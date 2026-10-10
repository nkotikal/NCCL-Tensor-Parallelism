# NCCL-Tensor-Parallelism
Tensor Parallelism implemented using NCCL for a hand-optimized GEMM in a PyTorch Transformer block with custom backend for causal FlashAttention in Triton. Validated and benchmarked against PyTorch single-GPU and TP standard implementations

Quick disclaimer, we don't truly save memory right now by running this because this is a pedagogical implementation of TP, so each rank still stores the full matrices and I call
ncclAllGather after every columnwise GEMM, which of course in a production implementation would not be used. 

Linux + CUDA + NCCL, Execution model: one process per GPU (`torch.multiprocessing.spawn`):


## Results (4xA100*-SXM4-40GB, fp32, B=2 N=16384 d=512 H=8)

| Path | ms / forward |
|---|---|
| TP ref (cuBLAS + `torch.distributed` + Triton attn) | 88.16 |
| TP custom (myGEMM + NCCL bind + Triton attn) | 89.94 |

\**Grabbed a node off of Vast.ai*



## NCCL performance analysis & Deriving bus bandwidth from the cost model

Let's begin by considering the ncclAllReduce ran on the 4xA100 node and assuming a standard ring model (which I believe nccl-tests does for busbw):

$$\text{Per-GPU }T_{comm} = 2(p-1)(\alpha + \frac{S}{pB})$$

Where S = message size, p = #GPUs and B = bandwidth of slowest link in the ring.
In `results/results_per_op.txt`, we see that the allReduce operation took 25.616ms. Treating latency as negligible (we can because of the massive message size) and using p = 4 and

$$S = 4\text{ bytes} \cdot batches \cdot N \cdot d = 4\cdot 2 \cdot 16384 \cdot 512 = 67.1 \text{MB (exactly 64 MiB)}$$

We can rearrange to get

$$B = \frac{2(p-1)\cdot S}{p\cdot T_{comm}} = \frac{6 \cdot 67.1}{4\cdot 0.025616} = 3929.18 MB/s = 3.93 GB/s$$

*(Verified with allGather at 12.901 ms results in 3.901 GB/s, approximates same busbw)*

Pretty slow... shouldn't NVLink bandwidth be in the hundreds of GB/s?

We can get a hint when looking at the adjacency list in `results/gpu_link_stats.txt` from running `nvidia-smi topo -m`:
```bash
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NV12    NODE    NODE 
GPU1    NV12     X      NODE    NODE
GPU2    NODE    NODE     X      NV10
GPU3    NODE    NODE    NV10     X  
```
Of the 4 GPUs in my node, NVLink is only connected for GPUs 0<->1 and 2<->3. In ring reduce, the bandwidth of the slowest link is the limiting bandwidth of the whole op. 

Since the slowest link is marked "NODE," we know that the path between the other GPUs goes through a PCIe CPU host bridge. Additionally, we know from the topo -p2p output that GPU Direct P2P is available between all the GPUs as well, including ones without NVLink. 

By running main.py after `export NCCL_DEBUG=INFO`, we get this block of lines:
```bash
dbaaa2700fd2:1893:1893 [3] NCCL INFO Channel 02 : 3[3] -> 0[0] via SHM/direct/direct
dbaaa2700fd2:1891:1891 [1] NCCL INFO Channel 02 : 1[1] -> 2[2] via SHM/direct/direct
dbaaa2700fd2:1892:1892 [2] NCCL INFO Channel 00/0 : 2[2] -> 3[3] via P2P/CUMEM/read
dbaaa2700fd2:1890:1890 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
dbaaa2700fd2:1890:1890 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/CUMEM/read
```

Between GPUs 2<->3 and 0<->1 which have NVLink, NCCL chooses to use P2P. However, for all other connections,
it chooses to go through PCIe SHM (shared memory, I believe this is not P2P but instead goes through host), implying that SHM could be better than the non-NVLink P2P on this node.

I tried proving this by running `profile_ops.py` with `export NCCL_P2P_LEVEL=SYS`. I thought this might allow P2P use instead of host bridge regardless of whether NVLink is available. However, the process hangs after the graph is created and I'm not able to get around it. Debug output and nvidia-smi revealed 100% utilization but relatively low power draw at around 50W/400W, and the P2P/CUMEM desired paths (such as 1[1] -> 2[2] were indeed being used, but I was unable to find the source of the hang as of yet. 

So instead, I created `link_bandwidth.py` which measures the P2P bandwidth between every GPU, and as we can see in `results/P2P_bandwidth.txt`, the peak bandwidth of the NVLink nodes is an exceptional 274 and 230 GB/s, while the other nodes have a dismal 2.7 GB/s. We can see that this figure, which is the *peak* of non-NVLink P2P bandwidth on this node, can't even compete with SHM. It is clear why NCCL chose the SHM/PCIe route (especially considering that the other one just hangs).



## Some other notes:
- Triton attn ~3x faster than fp32 SDPA (9.8 vs 30.1 ms, kind of a trivial result but also interesting when starting from fp32 SDPA as a baseline).
- NCCL bind matches `torch.distributed` (all_gather 12.9 ms, all_reduce 25.616 ms). See `results/results_per_op.txt`
- myGEMM ~8-9 TFLOPS vs cuBLAS ~13-14.5 (fp32 peak 19.5) -> ~2 ms gap.


## Run:
```bash
python main.py [-o out.txt] [-B 4] [-N 16384] [--reps 3]
python main.py --shapes -o <file.txt>
python profile_ops.py [-o ops.txt] [-B 2] [-N 16384]   # per-op timing (GEMMs, attention, collectives)
python kernels/benchmark.py -o <file.txt>
```

--------

## TODO:
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
