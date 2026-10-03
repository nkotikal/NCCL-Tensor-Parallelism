import torch
from torch import nn
from flash_attention_triton import __fwd_kernel_flash

class TransformerBlock(nn.Module):
    



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