# NCCL-Tensor-Parallelism
Optimized GEMM implemented under Tensor Parallelism using NCCL in a PyTorch Transformer block with custom backend for causal FlashAttention in Triton, validated and benchmarked against PyTorch single-GPU and TP standard implementations

Linux + CUDA + NCCL. One process per GPU (`torch.multiprocessing.spawn`):

```bash
python main.py           # correctness + benchmark → results.txt
python main.py --shapes  # also log per-GPU tensor shapes to results.txt
```

--------

### TODO:
- [x] Optimize GEMM and place the file in this workspace
- [x] Triton FlashAttention implementation
- [x] Pytorch Transformer block implementation
- [x] Add GELU gated by a bool to gemm epilogue for linear layer 1 of MLP
- [x] Hook up QKV projections and MLP block to my GEMM backend.
- [x] Split tensors to *n* GPUs
- [x] Use NCCL to coordinate the GEMM
- [ ] Benchmark against single-GPU and multi-GPU standard pytorch implementations
- [ ] Use TP for the attention launch as well
- [ ] Implement pipeline parallelism for n/2 GPU layers (n = # gpus) while keeping the other n/2 GPUs for TP
- [ ] Benchmark full suite

I used AI used for bindings, planning, and structuring benchmarks (essentially for scaffolding). 