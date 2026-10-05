# NCCL-Tensor-Parallelism
Optimized GEMM implemented under Tensor Parallelism using NCCL in a PyTorch Transformer block with custom backend for causal FlashAttention in Triton, validated and benchmarked against PyTorch single-GPU and TP standard implementations

Quick disclaimer, we don't truly save memory right now by running this because this is a pedagogical implementation of TP, so each rank still stores the full matrices and I call
ncclAllGather after every columnwise GEMM, which of course in a production implementation would not be used. 

Linux + CUDA + NCCL, Execution model: one process per GPU (`torch.multiprocessing.spawn`):

```bash
python main.py [-o out.txt] [-B 4] [-N 16384] [--reps 3]
python main.py --shapes -o shapes.txt
python kernels/benchmark.py -o kernels/kernel_benchmarks_RTX5090.txt
```

### Results (4x A100*-SXM4-40GB, fp32, B=2 N=16384 d=512 H=8)

| Path | ms / forward |
|---|---|
| TP ref (cuBLAS + `torch.distributed` + Triton attn) | 88.16 |
| TP custom (myGEMM + NCCL bind + Triton attn) | 89.94 |

- Triton attn ~3x faster than fp32 SDPA (9.8 vs 30.1 ms, kind of a trivial result but also interesting when starting from fp32 SDPA as a baseline).
- NCCL bind matches `torch.distributed` (all_gather 12.9 ms, all_reduce 25.6 ms).
- myGEMM ~8-9 TFLOPS vs cuBLAS ~13-14.5 (fp32 peak 19.5) -> ~2 ms gap.
- Comm is ~77 of ~90 ms: GPU pairs 0-1 / 2-3 are NVLinked, cross-pair NCCL ring goes through host memory (~3.9 GB/s). See `gpu_link_stats.txt`.

\**Grabbed a node off of Vast.ai*

--------

### TODO:
- [x] Optimize GEMM and place the file in this workspace
- [x] Triton FlashAttention implementation
- [x] Pytorch Transformer block implementation
- [x] Add GELU gated by a bool to gemm epilogue for linear layer 1 of MLP
- [x] Hook up QKV projections and MLP block to my GEMM backend.
- [x] Split tensors to *n* GPUs
- [x] Use NCCL to coordinate the GEMM
- [x] Benchmark against single-GPU and multi-GPU standard pytorch implementations

#### Future work:
- [ ] Profile with NSYS to verify bottlenecks that I discussed earlier
- [ ] Use TP for the attention launch as well
- [ ] Implement pipeline parallelism for n/2 GPU layers (n = # gpus) while keeping the other n/2 GPUs for TP
- [ ] Eventually remove extra all gathers and fully isolate GPU-specific tensor slices so we ACTUALLY save 
- [ ] Switch to bf16/fp16 across the repo to utilize tensor cores and compare with production level NCCL and CUDA
- [ ] Benchmark full suite

I used AI for bindings, planning, and structuring benchmarks (essentially for scaffolding). 