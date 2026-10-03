import triton
import triton.language as tl


#learned triton as I wrote this, so spammed comments for learning purposes. 
@triton.jit
def __fwd_kernel_flash(Q, K, V, O, N, inv_sq_d, 
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
    )