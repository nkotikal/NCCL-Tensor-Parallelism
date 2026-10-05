

from functools import cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent


@cache
def _ext():
    import sys
    import torch
    from torch.utils.cpp_extension import load

    if sys.platform == "win32":
        raise RuntimeError("NCCL needs Linux + CUDA")

    lib = Path(torch.__file__).resolve().parent / "lib"
    ld = ["-lnccl"]
    if lib.is_dir():
        ld += [f"-L{lib}", f"-Wl,-rpath,{lib}"]
    return load(name="my_nccl", sources=[str(_DIR / "bind_nccl.cpp")], extra_ldflags=ld)


def get_unique_id() -> bytes:
    return _ext().get_unique_id()


def init(rank: int, world_size: int, unique_id: bytes) -> None:
    _ext().init(rank, world_size, unique_id)


def destroy() -> None:
    _ext().destroy()


def all_reduce_sum(tensor) -> None:
    _ext().all_reduce_sum(tensor)


def all_gather(input_tensor, output_tensor) -> None:
    _ext().all_gather(input_tensor, output_tensor)
