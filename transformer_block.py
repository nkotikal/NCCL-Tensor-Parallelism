import torch
from torch import nn
from kernels.flash_attention_triton import __fwd_kernel_flash as forward_attention
import torch.nn.functional as F
from kernels.load_gemm import gemm as myGEMM

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
    
    def forward(self, X):
        if self.custom_GEMM:
            # gemm: C = A @ B with A (M,K), B (K,N), C (M,N); nn.Linear weight is (out, in) -> B = W^T
            batch_size, seq_len, in_features = X.shape
            out_features = self.weight.shape[0]
            m = batch_size * seq_len
            a = X.reshape(m, in_features).contiguous()
            b = self.weight.t().contiguous()
            c = torch.empty(m, out_features, device=X.device, dtype=X.dtype)
            myGEMM(a, b, c, 1, 0, self.GELU)
            return c.view(batch_size, seq_len, out_features)
        else:
            out = self.linear(X)
            if self.GELU:
                return F.gelu(out, approximate="tanh")
            return out
        

class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, TP_degree=8, custom_GEMM=False):
        super().__init__()
        assert d_model % heads == 0, "head sizes must be equal"
        #also d_model in our case should be divisible by 8 to make the 8-way TP possible, and likewise for any other TP degree.
        assert d_model % TP_degree == 0, f"d_model must be divisible by TP degree {TP_degree}"

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
    


    #TEST UNIT. This is my primary tensor parallel implementation of decoder block forward pass. 
    def forward_tensor_parallel(self, X):
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

        X = forward_attention(Q, K, V, causal=True)

        #Transpose and concatenate heads back to [B, N, d_model]
        X = X.transpose(1,2).contiguous().view(B, N, self.d_model)

        #O projection and add back residual
        X = self.wo(X)
        X = X + residual

        #MLP block
        residual = X
        X = self.ln2(X)
        X = self.mlp_1(X)
        X = self.mlp_2(X)
        X = X + residual

        return X

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
        X = self.mlp_1(X)
        X = self.mlp_2(X)
        X = X + residual

        return X
