
import os
from functools import cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def _cuda_include_dirs():
    """bind_nccl.cpp pulls in ATen CUDA headers; .cpp-only JIT does not add CUDA -I by default."""
    from torch.utils.cpp_extension import CUDA_HOME

    roots = []
    if CUDA_HOME:
        roots.append(Path(CUDA_HOME))
    env = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env:
        roots.append(Path(env))
    roots += [Path("/usr/local/cuda"), Path("/opt/cuda")]
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        inc = root / "include"
        if (inc / "cuda_runtime_api.h").exists():
            return [str(inc)]
    raise RuntimeError(
        "CUDA toolkit headers not found (cuda_runtime_api.h). "
        "Install cuda-toolkit or set CUDA_HOME, e.g. export CUDA_HOME=/usr/local/cuda"
    )


@cache
def _ext():
    import sys
    import torch
    from torch.utils.cpp_extension import load

    if sys.platform == "win32":
        raise RuntimeError("NCCL needs Linux + CUDA")

    inc = _cuda_include_dirs()
    extra_cflags = [f"-I{d}" for d in inc]

    lib = Path(torch.__file__).resolve().parent / "lib"
    ld = ["-lnccl"]
    if lib.is_dir():
        ld += [f"-L{lib}", f"-Wl,-rpath,{lib}"]
    return load(
        name="my_nccl",
        sources=[str(_DIR / "bind_nccl.cpp")],
        extra_cflags=extra_cflags,
        extra_ldflags=ld,
    )


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
