import torch
from torch import nn
from kernels.flash_attention_triton import __fwd_kernel_flash as forward_attention
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, TP_degree=8):
        super().__init__()
        assert d_model % heads == 0, "head sizes must be equal"
        #also d_model in our case should be divisible by 8 to make the 8-way TP possible, and likewise for any other TP degree.
        assert d_model % TP_degree == 0, f"d_model must be divisible by TP degree {TP_degree}"

        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        self.ln1 = nn.LayerNorm(d_model)

        #QKV and O projection matrices
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.ln2 = nn.LayerNorm(d_model)

        #MLP layers. 4x seems to be standard for up projection. Not using biases. 
        self.mlp_1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.mlp_2 = nn.Linear(4 * d_model, d_model, bias=False) 
    
    def forward(self, X):
        B, N, _ = X.shape #Batch size, sequence length, d_model


        residual = X


        #QKV projection, LayerNorm
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
        X = F.gelu(X)
        X = self.mlp_2(X)

        X = X + residual

        return X




