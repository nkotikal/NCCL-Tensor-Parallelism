# python profile_ops.py [-o ops.txt] [-B batch] [-N seq] [--reps N]
#
# Per-op timing of one TP forward: each GEMM, attention, and collective timed in isolation
# at the same per-rank shapes the block uses. Rank 0 wall time, ms per call.

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from nccl.load_nccl import all_gather, all_reduce_sum, destroy, get_unique_id
from nccl.load_nccl import init as initialize_nccl
from transformer_block import (
    ReferenceTransformerBlock,
    TransformerBlock,
    clear_results,
    log_result,
    set_results_path,
)

D, H = 512, 8


def timed(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / reps * 1e3
    dist.barrier()
    return ms


def run(rank, world, uid, output_path, reps, batch, seq_len):
    set_results_path(output_path)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=rank)
    initialize_nccl(rank, world, uid)

    from kernels.flash_attention_triton import __fwd_kernel_flash as triton_attn
    from kernels.load_gemm import gemm

    M = batch * seq_len
    s = D // world

    # per-rank shapes the block uses
    x = torch.randn(M, D, device=dev)
    w_cp = torch.randn(s, D, device=dev)
    w_m1 = torch.randn(4 * D // world, D, device=dev)
    w_m2 = torch.randn(D, 4 * D // world, device=dev)
    h_m2 = torch.randn(M, 4 * D // world, device=dev)

    def heads():
        return torch.randn(batch, seq_len, D, device=dev).view(batch, seq_len, H, D // H).transpose(1, 2)

    q, k, v = heads(), heads(), heads()
    shard = torch.randn(batch, seq_len, s, device=dev)
    gathered = torch.empty(world, batch, seq_len, s, device=dev)
    full = torch.randn(batch, seq_len, D, device=dev)

    def my_linear(a, w, gelu=False):
        c = torch.empty(a.shape[0], w.shape[0], device=dev)
        gemm(a, w.t().contiguous(), c, 1, 0, gelu)
        return c

    torch.manual_seed(0)
    ref = ReferenceTransformerBlock(D, H, TP_degree=world, tp_rank=rank).to(dev).eval()
    custom = TransformerBlock(D, H, TP_degree=world, tp_rank=rank, custom_GEMM=True).to(dev).eval()
    custom.load_state_dict(ref.state_dict())
    x3d = x.view(batch, seq_len, D)

    # ops shared by both TP paths
    t_cat = timed(lambda: torch.cat(list(gathered), dim=-1), reps)
    t_attn = timed(lambda: triton_attn(q, k, v, causal=True), reps)

    # TP custom: myGEMM + NCCL bind + Triton attention
    custom_ops = {
        "gemm QKV/wo shard": timed(lambda: my_linear(x, w_cp), reps),
        "gemm mlp_1 + GELU (fused)": timed(lambda: my_linear(x, w_m1, True), reps),
        "gemm mlp_2 shard": timed(lambda: my_linear(h_m2, w_m2), reps),
        "attention (Triton)": t_attn,
        "all_gather 1 shard": timed(lambda: all_gather(shard, gathered), reps),
        "all_reduce mlp_2": timed(lambda: all_reduce_sum(full), reps),
        "cat after gather": t_cat,
    }
    with torch.no_grad():
        custom_full = timed(lambda: custom.forward_tensor_parallel(x3d), reps)

    # TP ref: cuBLAS + torch.distributed + Triton attention
    ref_ops = {
        "gemm QKV/wo shard": timed(lambda: F.linear(x, w_cp), reps),
        "gemm mlp_1 + GELU": timed(lambda: F.gelu(F.linear(x, w_m1), approximate="tanh"), reps),
        "gemm mlp_2 shard": timed(lambda: F.linear(h_m2, w_m2), reps),
        "attention (Triton)": t_attn,
        "all_gather 1 shard": timed(lambda: dist.all_gather_into_tensor(gathered, shard), reps),
        "all_reduce mlp_2": timed(lambda: dist.all_reduce(full), reps),
        "cat after gather": t_cat,
    }
    with torch.no_grad():
        ref_full = timed(lambda: ref.forward_tensor_parallel_reference(x3d), reps)

    # single-GPU reference attention, for comparison only
    t_sdpa = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), reps)

    if rank == 0:
        nbytes = gathered.numel() * gathered.element_size()

        def report(title, ops, full_ms):
            # one forward: 4 column GEMMs + gathers + cats (Q, K, V, wo), mlp_1, mlp_2, attention, 1 all_reduce
            vals = list(ops.values())
            cp, m1, m2, attn, ag, ar, cat = vals
            parts = 4 * cp + m1 + m2 + attn + 4 * ag + 4 * cat + ar
            gather_bw = nbytes * (world - 1) / world / (ag / 1e3) / 1e9
            reduce_bw = full.numel() * full.element_size() * 2 * (world - 1) / world / (ar / 1e3) / 1e9
            log_result(f"[{title}]")
            for name, ms in ops.items():
                log_result(f"  {name:<28} {ms:8.3f}")
            log_result(f"  {'sum of parts per forward':<28} {parts:8.3f}")
            log_result(f"  {'full forward (measured)':<28} {full_ms:8.3f}")
            log_result(f"  busBW all_gather {gather_bw:6.1f} GB/s, all_reduce {reduce_bw:6.1f} GB/s")
            log_result("")

        log_result(f"B={batch} N={seq_len} d={D} H={H} world={world}  (ms per call, rank 0)")
        log_result("")
        report("TP custom: myGEMM + NCCL bind + Triton attn", custom_ops, custom_full)
        report("TP ref: cuBLAS + torch.distributed + Triton attn", ref_ops, ref_full)
        log_result("[single-GPU reference attention]")
        log_result(f"  {'attention (SDPA)':<28} {t_sdpa:8.3f}")

    destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-o", "--output", default="results/results_per_op.txt",
        help="per-op timing log (default: results/results_per_op.txt)",
    )
    p.add_argument("--reps", type=int, default=10, help="timed iterations per op (default: 10)")
    p.add_argument("-B", "--batch", type=int, default=2, help="batch size (default: 2)")
    p.add_argument("-N", "--seq-len", type=int, default=16384, help="sequence length (default: 16384)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    set_results_path(args.output)
    clear_results()
    world = torch.cuda.device_count()
    uid = get_unique_id()
    mp.spawn(
        run,
        args=(world, uid, args.output, args.reps, args.batch, args.seq_len),
        nprocs=world,
        join=True,
    )
