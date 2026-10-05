# python main.py [-o out.txt] [--shapes]
#
# mp.spawn = start N fresh Python workers (one per GPU). CUDA needs spawn, not threads.
# Same idea as torchrun; you could use Process(...).start() in a loop instead.

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nccl.load_nccl import destroy, get_unique_id
from nccl.load_nccl import init as initialize_nccl
from transformer_block import (
    ReferenceTransformerBlock,
    TransformerBlock,
    clear_results,
    log_result,
    set_results_path,
)

# benchmark shape
B, N, D, H = 1, 8192, 512, 8
# Triton attn -> looser tol than SDPA
ATOL_SDPA = 1e-3
ATOL_TRITON = 1e-2


def _progress(rank, msg):
    if rank == 0:
        print(msg, file=sys.stderr, flush=True)


def run(rank, world, uid, log_shapes, output_path):
    set_results_path(output_path)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    # dist: forward_tensor_parallel_reference
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=rank)

    # bind_nccl init: forward_tensor_parallel
    initialize_nccl(rank, world, uid)
    _progress(rank, "[rank 0] NCCL extension ready")

    # JIT on every rank before any custom-path collective (else fast ranks hang in all_gather)
    from kernels.flash_attention_triton import __fwd_kernel_flash as forward_attention
    from kernels.load_gemm import _module as load_gemm_module, gemm

    _progress(rank, "[rank 0] JIT compiling myGEMM (all ranks; can take several minutes)...")
    load_gemm_module()
    m, k, n = 64, 64, 64
    a = torch.randn(m, k, device=dev)
    b = torch.randn(k, n, device=dev)
    c = torch.zeros(m, n, device=dev)
    gemm(a, b, c)
    torch.cuda.synchronize()
    dist.barrier()
    _progress(rank, "[rank 0] myGEMM ready; JIT compiling Triton attention...")
    hw, nw, dw = H, 64, D // H
    q = torch.randn(1, hw, nw, dw, device=dev)
    forward_attention(q, q, q, causal=True)
    torch.cuda.synchronize()
    dist.barrier()
    _progress(rank, "[rank 0] JIT warmup done; running forwards...")

    # shared weights (ref nn.Linear -> custom myGEMM)
    torch.manual_seed(0)
    ref = ReferenceTransformerBlock(D, H, TP_degree=world, tp_rank=rank).to(dev).eval()
    custom = TransformerBlock(D, H, TP_degree=world, tp_rank=rank, custom_GEMM=True).to(dev).eval()
    custom.load_state_dict(ref.state_dict())

    # same X on all ranks
    if rank == 0:
        x = torch.randn(B, N, D, device=dev)
    else:
        x = torch.empty(B, N, D, device=dev)
    dist.broadcast(x, 0)

    # three forwards, compare on rank 0
    with torch.no_grad():
        y_single = ref.forward_single_GPU_reference(x) if rank == 0 else None
        y_tp_ref = ref.forward_tensor_parallel_reference(x, log_shapes=log_shapes)
        y_custom = custom.forward_tensor_parallel(x, log_shapes=log_shapes)

    if log_shapes:
        dist.barrier()

    if rank == 0:
        torch.testing.assert_close(y_tp_ref, y_single, atol=ATOL_SDPA, rtol=ATOL_SDPA)
        torch.testing.assert_close(y_custom, y_tp_ref, atol=ATOL_TRITON, rtol=ATOL_TRITON)
        log_result("correctness: single vs tp_ref (SDPA), custom vs tp_ref (Triton attn) ok")

    dist.barrier()
    _progress(rank, "[rank 0] correctness ok; benchmarking...")

    # cuda timer + barrier for TP
    def ms(fn, reps=5):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        return (time.perf_counter() - t0) / reps * 1e3

    with torch.no_grad():
        t_single = ms(lambda: ref.forward_single_GPU_reference(x)) if rank == 0 else 0.0
        t_tp_ref = ms(lambda: ref.forward_tensor_parallel_reference(x))
        t_custom = ms(lambda: custom.forward_tensor_parallel(x))

    if rank == 0:
        log_result(f"B={B} N={N} d={D} world={world}  (ms per forward, rank 0)")
        log_result(f"  single_GPU_reference (SDPA):     {t_single:.2f}")
        log_result(f"  tensor_parallel_reference:       {t_tp_ref:.2f}")
        log_result(f"  tensor_parallel (GEMM+Triton):   {t_custom:.2f}")

    destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", action="store_true", help="log per-GPU shard/gather shapes for both TP paths")
    p.add_argument("-o", "--output", default="results.txt", help="benchmark/shape log file (default: results.txt)")
    args = p.parse_args()

    set_results_path(args.output)
    clear_results()
    world = torch.cuda.device_count()
    uid = get_unique_id()
    mp.spawn(run, args=(world, uid, args.shapes, args.output), nprocs=world, join=True)
