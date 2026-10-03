"""
Benchmark custom kernels vs PyTorch. Run from repo root:

  python kernels/benchmark.py --gemm-m 4096 --gemm-k 4096 --gemm-n 4096
  python kernels/benchmark.py --attn-b 4 --attn-h 16 --attn-n 1024 --attn-d 64
  python kernels/benchmark.py   # default gemm + attention
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kernels.flash_attention_triton import __fwd_kernel_flash as flash_attn
from kernels.load_gemm import gemm


def bench_ms(fn):
    return triton.testing.do_bench(fn)


def bench_gemm(M, K, N, gelu=False):
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    def mine():
        gemm(A, B, C, 1.0, 0.0, gelu)

    def torch_ref():
        out = A @ B
        if gelu:
            out = F.gelu(out, approximate="tanh")
        return out

    # warmup compile for custom gemm on first call
    mine()
    t_custom = bench_ms(mine)
    t_torch = bench_ms(torch_ref)
    print(f"GEMM  M={M} K={K} N={N}  gelu={gelu}")
    print(f"  myGEMM     {t_custom:.3f} ms")
    print(f"  torch.matmul {t_torch:.3f} ms  ({t_torch / t_custom:.2f}x vs custom)")


def bench_attention(B, H, N, d, causal=True):
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

    flash_attn(q, k, v, causal=causal)
    t_custom = bench_ms(lambda: flash_attn(q, k, v, causal=causal))
    t_sdpa = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
    print(f"Attention  B={B} H={H} N={N} d={d}  causal={causal}")
    print(f"  Triton flash {t_custom:.3f} ms")
    print(f"  SDPA         {t_sdpa:.3f} ms  ({t_sdpa / t_custom:.2f}x vs Triton)")


def main():
    p = argparse.ArgumentParser(description="Benchmark kernels vs PyTorch")
    p.add_argument("--gemm-m", type=int, default=2048)
    p.add_argument("--gemm-k", type=int, default=2048)
    p.add_argument("--gemm-n", type=int, default=2048)
    p.add_argument("--gemm-gelu", action="store_true")
    p.add_argument("--attn-b", type=int, default=2)
    p.add_argument("--attn-h", type=int, default=8)
    p.add_argument("--attn-n", type=int, default=1024, help="sequence length")
    p.add_argument("--attn-d", type=int, default=64, help="head dim (power of 2)")
    p.add_argument("--no-causal", action="store_true")
    p.add_argument("--gemm-only", action="store_true")
    p.add_argument("--attn-only", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    causal = not args.no_causal
    run_gemm = not args.attn_only
    run_attn = not args.gemm_only

    if run_gemm:
        bench_gemm(args.gemm_m, args.gemm_k, args.gemm_n, args.gemm_gelu)
    if run_attn:
        if run_gemm:
            print()
        bench_attention(args.attn_b, args.attn_h, args.attn_n, args.attn_d, causal)


if __name__ == "__main__":
    main()
