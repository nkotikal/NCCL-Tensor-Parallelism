import time

import torch

NBYTES = 256 * 1024 * 1024
REPS = 20


def bw(copy_fn, sync_devs): #formula: B = S / (copy time) / 1e9
    for _ in range(3):
        copy_fn()
    for d in sync_devs:
        torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for _ in range(REPS):
        copy_fn()
    for d in sync_devs:
        torch.cuda.synchronize(d)
    return NBYTES * REPS / (time.perf_counter() - t0) / 1e9


n = torch.cuda.device_count()
print(f"copy size {NBYTES >> 20} MiB, {REPS} reps")

print("host <-> device (pinned), GB/s")
for g in range(n):
    host = torch.empty(NBYTES, dtype=torch.uint8).pin_memory()
    dev = torch.empty(NBYTES, dtype=torch.uint8, device=f"cuda:{g}")
    h2d = bw(lambda: dev.copy_(host, non_blocking=True), [g])
    d2h = bw(lambda: host.copy_(dev, non_blocking=True), [g])
    print(f"  GPU{g}  H2D {h2d:6.1f}  D2H {d2h:6.1f}")

print("GPU -> GPU, GB/s (peer access shown)")
for src, dst in [(0, 1), (2, 3), (1, 2), (3, 0), (0, 2)]:
    a = torch.empty(NBYTES, dtype=torch.uint8, device=f"cuda:{src}")
    b = torch.empty(NBYTES, dtype=torch.uint8, device=f"cuda:{dst}")
    peer = torch.cuda.can_device_access_peer(src, dst)
    r = bw(lambda: b.copy_(a, non_blocking=True), [src, dst])
    print(f"  {src} -> {dst}  peer={peer}  {r:6.1f}")
