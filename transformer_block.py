import torch
from torch import nn
from kernels.flash_attention_triton import __fwd_kernel_flash as forward_attention
import torch.nn.functional as F
from kernels.load_gemm import gemm as myGEMM
from nccl.load_nccl import all_gather, all_reduce_sum

class myTPLinear(nn.Module):
    def __init__(self, in_features, out_features, custom_GEMM=False, GELU=False):
        super().__init__()
        self.custom_GEMM = custom_GEMM
        self.GELU = GELU
        if self.custom_GEMM:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, X, rank=0, tp_degree=1, row_parallel=False):
        W = self.weight if self.custom_GEMM else self.linear.weight
        if row_parallel:
            cols = W.shape[1] // tp_degree
            c0 = rank * cols
            W = W[:, c0 : c0 + cols]
            if X.shape[-1] != cols:
                X = X[..., c0 : c0 + cols]
        else:
            rows = W.shape[0] // tp_degree
            W = W[rank * rows : (rank + 1) * rows]
        if self.custom_GEMM:
            # gemm: C = A @ B with A (M,K), B (K,N), C (M,N); nn.Linear weight is (out, in) -> B = W^T
            batch_size, seq_len, in_features = X.shape
            out_features = W.shape[0]
            m = batch_size * seq_len
            a = X.reshape(m, in_features).contiguous()
            b = W.t().contiguous()
            c = torch.empty(m, out_features, device=X.device, dtype=X.dtype)
            myGEMM(a, b, c, 1, 0, self.GELU)
            return c.view(batch_size, seq_len, out_features)
        else:
            out = F.linear(X, W)
            if self.GELU:
                return F.gelu(out, approximate="tanh")
            return out
        

class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, TP_degree=8, tp_rank=0, custom_GEMM=False):
        super().__init__()
        assert d_model % heads == 0, "head sizes must be equal"
        #also d_model in our case should be divisible by 8 to make the 8-way TP possible, and likewise for any other TP degree.
        assert d_model % TP_degree == 0, f"d_model must be divisible by TP degree {TP_degree}"

        self.TP_degree = TP_degree
        self.tp_rank = tp_rank
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.custom_GEMM = custom_GEMM

        self.ln1 = nn.LayerNorm(d_model)

        # QKV and O projection matrices
        self.wq = myTPLinear(d_model, d_model, custom_GEMM=custom_GEMM)
        self.wk = myTPLinear(d_model, d_model, custom_GEMM=custom_GEMM)
        self.wv = myTPLinear(d_model, d_model, custom_GEMM=custom_GEMM)
        self.wo = myTPLinear(d_model, d_model, custom_GEMM=custom_GEMM)

        self.ln2 = nn.LayerNorm(d_model)

        #MLP layers. 4x seems to be standard for up projection. Not using biases. 
        self.mlp_1 = myTPLinear(d_model, 4 * d_model, custom_GEMM=custom_GEMM, GELU=True)
        self.mlp_2 = myTPLinear(4 * d_model, d_model, custom_GEMM=custom_GEMM) 
    


    # Column-parallel GEMM on this rank, then all_gather the output shards back to [B, N, out].
    def tp(self, tp_linear, X):
        shard = tp_linear(X, self.tp_rank, self.TP_degree)
        B, N, s = shard.shape
        # ncclAllGather stacks rank buffers along the first dim: out[r] is rank r's shard
        out = torch.empty(self.TP_degree, B, N, s, device=X.device, dtype=X.dtype)
        all_gather(shard.contiguous(), out)
        return torch.cat(list(out), dim=-1)

    #TEST UNIT. This is my primary tensor parallel implementation of decoder block forward pass. 
    def forward_tensor_parallel(self, X):
        B, N, _ = X.shape #Batch size, sequence length, d_model


        residual = X



        #LayerNorm, QKV projection
        X = self.ln1(X)

        Q = self.tp(self.wq, X)
        K = self.tp(self.wk, X)
        V = self.tp(self.wv, X)
        
        #Split into heads and transpose to get the shape [B, heads, N, head_dim] for triton MHA.
        Q = Q.view(B, N, self.heads, self.head_dim).transpose(1,2)
        K = K.view(B, N, self.heads, self.head_dim).transpose(1,2)
        V = V.view(B, N, self.heads, self.head_dim).transpose(1,2)

        X = forward_attention(Q, K, V, causal=True)

        #Transpose and concatenate heads back to [B, N, d_model]
        X = X.transpose(1,2).contiguous().view(B, N, self.d_model)

        #O projection and add back residual (full linear; attention already full [B,N,d])
        X = self.wo(X)
        X = X + residual

        #MLP block
        residual = X
        X = self.ln2(X)
        X = self.mlp_1(X, self.tp_rank, self.TP_degree)
        X = self.mlp_2(X, self.tp_rank, self.TP_degree, row_parallel=True)
        all_reduce_sum(X)
        X = X + residual

        return X

    def forward_single_GPU_reference(self, X): #must also initialize with custom_GEMM=False
        B, N, _ = X.shape #Batch size, sequence length, d_model


        residual = X


        #LayerNorm, QKV projection
        X = self.ln1(X)

        Q = self.wq(X)
        K = self.wk(X)
        V = self.wv(X)
        
        #Split into heads and transpose to get the shape [B, heads, N, head_dim] for triton MHA.
        Q = Q.view(B, N, self.heads, self.head_dim).transpose(1,2)
        K = K.view(B, N, self.heads, self.head_dim).transpose(1,2)
        V = V.view(B, N, self.heads, self.head_dim).transpose(1,2)


        X = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        X = X.transpose(1, 2).contiguous().view(B, N, self.d_model)
        X = self.wo(X)
        X = X + residual

        residual = X
        X = self.ln2(X)
        X = self.mlp_1(X)
        X = self.mlp_2(X)
        X = X + residual

        return X
