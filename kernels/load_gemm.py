from functools import cache
from pathlib import Path

from torch.utils.cpp_extension import load

_DIR = Path(__file__).resolve().parent


@cache
def _module():
    return load(
        name="my_gemm",
        sources=[str(_DIR / "bind_gemm.cpp"), str(_DIR / "myGEMM.cu")],
        extra_cuda_cflags=["-O3"],
    )


def gemm(A, B, C, alpha=1.0, beta=0.0, gelu=False):
    """
    In-place GEMM: C = alpha * (A @ B) + beta * C, optional GELU on C.
    A: (M, K), B: (K, N), C: (M, N), float32, CUDA, contiguous.
    """
    return _module().gemm(A, B, C, alpha, beta, gelu)
