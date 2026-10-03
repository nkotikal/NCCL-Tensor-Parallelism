import torch
from torch import nn
from flash_attention_triton import __fwd_kernel_flash

class TransformerBlock(nn.Module):
    


