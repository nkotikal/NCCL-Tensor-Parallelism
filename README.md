# NCCL-Tensor-Parallelism
Optimized GEMM implemented under Tensor Parallelism using NCCL in a Transformer block implemented with FlashAttention in Pytorch, validated and benchmarked against PyTorch standard implementations

\<In progress>

--------

### TODO:
- [x] Optimize GEMM and place the file in this workspace
- [ ] PyTorch FlashAttention implementation
- [ ] Add GELU gated by a bool to gemm epilogue for linear layer 1 of MLP
- [ ] Hook up QKV projections and MLP block to my GEMM backend.
- [ ] Split tensors to 8 GPUs
- [ ] Use NCCL to coordinate the GEMM
- [ ] Benchmark against single-GPU and multi-GPU standard pytorch implementations