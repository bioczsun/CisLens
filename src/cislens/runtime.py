"""Seeding, atomic outputs and recorded execution environments."""

from contextlib import contextmanager
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import tempfile

import numpy as np
import torch


def seed_everything(seed, strict=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(strict)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(value):
    device = torch.device(
        "cuda"
        if value == "auto" and torch.cuda.is_available()
        else "cpu"
        if value == "auto"
        else value
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; select --device cpu or install a matching PyTorch/CUDA build"
        )
    return device


def environment():
    from . import __version__

    versions = {}
    for name in (
        "torch",
        "numpy",
        "pandas",
        "h5py",
        "scipy",
        "scikit-image",
        "einops",
        "pyBigWig",
        "pyfaidx",
        "cooler",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {
        "cislens": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "strict_determinism": torch.are_deterministic_algorithms_enabled(),
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@contextmanager
def atomic_output(path, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}; use an explicit overwrite option")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        if overwrite:
            temporary.replace(path)
        else:
            # Hard-link publication is atomic and refuses concurrent overwrite.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
