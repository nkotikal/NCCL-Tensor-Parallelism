"""JIT-load PyTorch extension that calls the NCCL C API (Linux + CUDA only)."""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def _nccl_link_flags() -> tuple[list[str], list[str]]:
    import torch

    if sys.platform == "win32":
        raise RuntimeError(
            "NCCL is not supported on Windows. Run collective tests on Linux/WSL or a multi-GPU pod."
        )
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    extra_ldflags = ["-lnccl"]
    if torch_lib.is_dir():
        extra_ldflags.append(f"-L{torch_lib}")
        extra_ldflags.append(f"-Wl,-rpath,{torch_lib}")
    return [], extra_ldflags


@cache
def _module():
    extra_cflags, extra_ldflags = _nccl_link_flags()
    from torch.utils.cpp_extension import load

    return load(
        name="my_nccl",
        sources=[str(_DIR / "bind_nccl.cpp")],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
    )


def get_unique_id() -> bytes:
    return _module().get_unique_id()


def init(rank: int, world_size: int, unique_id: bytes) -> None:
    _module().init(rank, world_size, unique_id)


def destroy() -> None:
    _module().destroy()


def rank() -> int:
    return _module().rank()


def world_size() -> int:
    return _module().world_size()


def all_reduce(tensor, redop: int = 0) -> None:
    """In-place sum (redop 0) by default."""
    _module().all_reduce(tensor, redop, True)


def all_reduce_out(input_tensor, output_tensor, redop: int = 0) -> None:
    _module().all_reduce_out(input_tensor, output_tensor, redop)


def broadcast(tensor, root: int = 0) -> None:
    _module().broadcast(tensor, root)


def all_gather(input_tensor, output_tensor) -> None:
    _module().all_gather(input_tensor, output_tensor)


def reduce_scatter(input_tensor, output_tensor) -> None:
    _module().reduce_scatter(input_tensor, output_tensor)


def group_all_reduce(a, b, redop: int = 0) -> None:
    _module().group_all_reduce(a, b, redop)
