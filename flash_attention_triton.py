import torch
import triton
import triton.language as tl



LOG2E = 1.4426950408889634  #log_2(e) for fast exp2 -> exp

#learned triton as I wrote this, so spammed comments for learning purposes. 
@triton.jit
def __fwd_kernel_flash_triton(Q, K, V, O, N, qk_scale, #qk scale is 1/sqrt(d) * log_2(e), factoring in the log_2 optimization with the head dim scale. 
                        Br: tl.constexpr, Bc: tl.constexpr, #standard that these should be known at compile time. 
                        d: tl.constexpr, causal_bool: tl.constexpr):
    #tl.program_id(axis) is analogous to blockIdx.axis in CUDA, but represents instances of the kernel
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    base = pid_bh * N * d #offset = (b*H + h)*N*d  + adding N*d + d when loading Q tile

    offs_m = pid_m * Br + tl.arange(0, Br) #Q tile load rows
    offs_n = tl.arange(0, Bc) #K/V tile load columns
    offs_d = tl.arange(0, d) #head dimension indices

    q = tl.load(
        #Q + base finds the offset row for our tile. Then we load in Br rows dictated by offs_m, each one offs_d columns. 
        Q + base + offs_m[:, None] * d + offs_d[None, :], #adding length one axis to offset, gets us to the proper row (dependent on offs_m)
        mask=offs_m[:, None] < N, 
        other=0.0 #default if mask is false (bounds check)
    )

    #online softmax states stored for each query row. 
    m_i = tl.full([Br], -float('inf'), tl.float32) #running max of scores
    l_i = tl.zeros([Br], tl.float32) #running sum of exp(score-m_i) for softmax denominator
    acc = tl.zeros([Br, d], tl.float32) #tracks softmax numerator, exp(score-m_i) * V (value). 

    if causal_bool: #create causal mask
        high = tl.minimum(((pid_m + 1) * Br), N) #max rows to load from K/V tiles to make it causal
    else:
        high = N
    
    #inner loop over K/V
    for start in range(0, high, Bc): #iterate over tiles by adding +Bc until mask limit
        cols = start + offs_n

        k = tl.load(
            K + base + cols[None, :] * d + offs_d[:, None], #K is transposed so we physically are still loading contiguous rows
            mask=cols[None, :] < N,
            other=0.0
        )

        s = tl.dot(q, k) * qk_scale #multiply the QK tiles with dims Br x d @ d x Bc to get a Br x Bc tile space of O matrix. 

        mask = cols[None, :] < N
        if causal_bool:
            #sets mask for anywhere where the attention weight matrix key index <= query index, masking out everything else. 
            mask = mask & (cols[None, :] <= offs_m[:, None]) 

        s = tl.where(mask, s, -float('inf')) #this is there the real masking happens. Doesnt contribute to softmax. 


        #time for the online softmax
        m_new = tl.maximum(m_i, tl.max(s, 1)) #row-wise max for the keys in the tile
        #e^x = 2^(x log_2(e)), givings us log_2(e) as a constant. 
        #Instead of exp, we use this because exp2 is faster on hardware.
        #due to log_2(e) being a constant, it's folded into qk_scale. 
        p = tl.math.exp2(s - m_new[:, None])
        adj = tl.math.exp2(m_i - m_new) #adjustment factor

        l_i = l_i * adj + tl.sum(p, 1) #need to sum for softmax denominator
        #online softmax math is confusing...

        v = tl.load(
            V + base + cols[:, None] * d + offs_d[None, :],
            mask=cols[:, None] < N,
            other=0.0
        )
        acc = acc * adj[:, None] + tl.dot(p, v)
        m_i = m_new
    
    #final division! divide all the attention weights by the softmax denominator.
    acc = acc / l_i[:, None]

    #store result in output
    tl.store(
        O + base + offs_m[:, None] * d + offs_d[None, :],
        acc,
        mask=offs_m[:, None] < N
    )

def __fwd_kernel_flash(Q, K, V, causal=False, Br=64, Bc=64):
    B, H, N, d = Q.shape
    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    O = torch.empty_like(q)
    grid = (triton.cdiv(N, Br), B * H)
    qk_scale = (d ** -0.5) * LOG2E #see top for LOG2E def
    __fwd_kernel_flash_triton[grid](Q, K, V, O, N, qk_scale, Br, Bc, d, casual_bool, num_warps=4, num_stages=2)
    #last two args are optional but like cuda kernel launch, but for triton.
    return O


"""
Flash attention algorithm flow:



load Q tile

m = -inf
l = 0
acc = 0

for K/V tile:
    load K
    load V

    scores = Q @ K.T
    scores *= scale

    apply causal mask

    # online softmax
    m_new = max(m, rowmax(scores))
    p = exp(scores - m_new)

    acc = acc * exp(m - m_new) + p @ V
    l   = l * exp(m - m_new) + sum(p)

    m = m_new

O = acc / l
store O
"""