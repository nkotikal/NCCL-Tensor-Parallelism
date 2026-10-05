import torch
from torch import nn
from kernels.flash_attention_triton import __fwd_kernel_flash as forward_attention
import torch.nn.functional as F
from kernels.load_gemm import gemm as myGEMM
from nccl.load_nccl import all_gather, all_reduce_sum

RESULTS_PATH = "results.txt"


def set_results_path(path: str) -> None:
    global RESULTS_PATH
    RESULTS_PATH = path


def log_result(msg: str) -> None:
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def clear_results() -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        pass


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
        #if we choose a row-parallel layout, we shard W by rows (columns in this case because of transposed). This means we also
        #shard X by columns (untransposed). 
        if row_parallel:
            cols = W.shape[1] // tp_degree # tp_degree "rows"
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
    


    # Main function of this is just so we can all_gather shards. I'm using if for qkv 
    def tp(self, tp_linear, X, tag="", log_shapes=False):
        shard = tp_linear(X, self.tp_rank, self.TP_degree)
        B, N, s = shard.shape
        if log_shapes:
            log_result(f"[custom gpu {self.tp_rank}] {tag} cp_shard {tuple(shard.shape)}")
        # ncclAllGather stacks rank buffers along the first dim: out[r] is rank r's shard
        out = torch.empty(self.TP_degree, B, N, s, device=X.device, dtype=X.dtype)
        all_gather(shard.contiguous(), out)
        full = torch.cat(list(out), dim=-1)
        if log_shapes:
            log_result(f"[custom gpu {self.tp_rank}] {tag} after_gather {tuple(full.shape)}")
        return full

    #TEST UNIT. This is my primary tensor parallel implementation of decoder block forward pass. 
    def forward_tensor_parallel(self, X, log_shapes=False):
        B, N, _ = X.shape #Batch size, sequence length, d_model


        residual = X



        #LayerNorm, QKV projection
        X = self.ln1(X)

        Q = self.tp(self.wq, X, "Q", log_shapes)
        K = self.tp(self.wk, X, "K", log_shapes)
        V = self.tp(self.wv, X, "V", log_shapes)
        
        #Split into heads and transpose to get the shape [B, heads, N, head_dim] for triton MHA.
        Q = Q.view(B, N, self.heads, self.head_dim).transpose(1,2)
        K = K.view(B, N, self.heads, self.head_dim).transpose(1,2)
        V = V.view(B, N, self.heads, self.head_dim).transpose(1,2)

        X = forward_attention(Q, K, V, causal=True)

        #Transpose and concatenate heads back to [B, N, d_model]
        X = X.transpose(1,2).contiguous().view(B, N, self.d_model)

        #O projection and add back residual
        X = self.tp(self.wo, X, "wo", log_shapes)
        X = X + residual

        #MLP block
        residual = X
        X = self.ln2(X)
        X = self.mlp_1(X, self.tp_rank, self.TP_degree)
        if log_shapes:
            log_result(f"[custom gpu {self.tp_rank}] mlp_1 cp_shard {tuple(X.shape)}")
        X = self.mlp_2(X, self.tp_rank, self.TP_degree, row_parallel=True)
        if log_shapes:
            log_result(f"[custom gpu {self.tp_rank}] mlp_2 rp_partial {tuple(X.shape)}")
        all_reduce_sum(X)
        if log_shapes:
            log_result(f"[custom gpu {self.tp_rank}] mlp_2 after_allreduce {tuple(X.shape)}")
        X = X + residual

        return X








# BELOW: Reference iplementations

import torch.distributed as dist

class ReferenceTransformerBlock(nn.Module):
    def __init__(self, d_model, heads, TP_degree=1, tp_rank=0):
        super().__init__()
        self.TP_degree = TP_degree
        self.tp_rank = tp_rank
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        self.ln1 = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.mlp_2 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward_single_GPU_reference(self, X):
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
        X = F.gelu(self.mlp_1(X), approximate="tanh")
        X = self.mlp_2(X)
        X = X + residual

        return X

    # Same TP layout as TransformerBlock.forward_tensor_parallel, using torch.distributed + Triton attention.
    def forward_tensor_parallel_reference(self, X, log_shapes=False):
        B, N, _ = X.shape
        r, tp = self.tp_rank, self.TP_degree

        def col_gather(lin, X, tag=""):
            shard = F.linear(X, lin.weight.chunk(tp)[r])
            if log_shapes:
                log_result(f"[tp_ref gpu {r}] {tag} cp_shard {tuple(shard.shape)}")
            out = torch.empty(tp, *shard.shape, device=X.device, dtype=X.dtype)
            dist.all_gather_into_tensor(out, shard.contiguous())
            full = torch.cat(list(out), dim=-1)
            if log_shapes:
                log_result(f"[tp_ref gpu {r}] {tag} after_gather {tuple(full.shape)}")
            return full

        residual = X
        X = self.ln1(X)
        Q = col_gather(self.wq, X, "Q").view(B, N, self.heads, self.head_dim).transpose(1, 2)
        K = col_gather(self.wk, X, "K").view(B, N, self.heads, self.head_dim).transpose(1, 2)
        V = col_gather(self.wv, X, "V").view(B, N, self.heads, self.head_dim).transpose(1, 2)
        X = forward_attention(Q, K, V, causal=True)
        X = X.transpose(1, 2).contiguous().view(B, N, self.d_model)
        X = col_gather(self.wo, X, "wo") + residual

        residual = X
        X = self.ln2(X)
        X = F.gelu(F.linear(X, self.mlp_1.weight.chunk(tp)[r]), approximate="tanh")
        if log_shapes:
            log_result(f"[tp_ref gpu {r}] mlp_1 cp_shard {tuple(X.shape)}")
        X = F.linear(X, self.mlp_2.weight.chunk(tp, dim=1)[r])
        if log_shapes:
            log_result(f"[tp_ref gpu {r}] mlp_2 rp_partial {tuple(X.shape)}")
        dist.all_reduce(X)
        if log_shapes:
            log_result(f"[tp_ref gpu {r}] mlp_2 after_allreduce {tuple(X.shape)}")
        return X + residual
