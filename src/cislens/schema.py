"""Shared data contract. Genomic coordinates are zero-based, half-open."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
BED_COLUMNS = ["chrom", "start", "end", "split"]
TARGET_NAMES = (
    "H3K27ac",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K9ac",
    "H3K36me3",
    "H2AFZ",
    "H3K27me3",
    "H3K9me3",
    "H4K20me1",
    "H3K79me2",
    "CHD1",
    "CTCF",
    "RAD21",
    "SMC3",
)


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 960
    qk_dim: int = 64
    v_dim: int = 64
    num_head: int = 8
    depth: int = 6
    pool_size: int = 4
    mid_dim: int = 128
    max_seq_len: int = 512
    seq_bins: int = 1024
    seq_res: int = 2048
    output_channels: int = 15

    def __post_init__(self):
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in asdict(self).values()
        ):
            raise ValueError("All model dimensions must be positive integers")
        if self.qk_dim % 2 or self.seq_bins % self.pool_size:
            raise ValueError("qk_dim must be even; seq_bins must be divisible by pool_size")
        if self.window < 128:
            raise ValueError("Signal windows must contain at least 128 bases")

    @property
    def window(self):
        return self.seq_bins * self.seq_res

    def build(self):
        from .network import CisLens

        options = asdict(self)
        for key in ("seq_bins", "seq_res", "output_channels"):
            options.pop(key)
        return CisLens(**options, output_heads={"epi": self.output_channels})


def read_bed(path, geometry):
    frame = pd.read_csv(path, sep="\t", header=None, comment="#", dtype=str, keep_default_na=False)
    if frame.empty or frame.shape[1] != 4:
        raise ValueError(f"Expected a nonempty four-column BED (chrom/start/end/split): {path}")
    frame.columns = BED_COLUMNS
    for column in ("start", "end"):
        if not frame[column].str.fullmatch(r"\d+").all():
            raise ValueError(f"Invalid nonnegative integer {column} in {path}")
        frame[column] = frame[column].astype(np.int64)
    if not frame.split.isin(SPLITS).all():
        raise ValueError(f"BED split must be train, val or test: {path}")
    if not ((frame.end - frame.start) == geometry.window).all():
        raise ValueError(f"BED windows must be {geometry.window} bp: {path}")
    if (frame.start % geometry.seq_res).any():
        raise ValueError(f"BED starts must align to {geometry.seq_res} bp: {path}")
    if frame.chrom.eq("").any() or frame.duplicated(BED_COLUMNS[:3]).any():
        raise ValueError(f"Empty chromosome or duplicate BED windows: {path}")
    if (frame.groupby("chrom").split.nunique() > 1).any():
        raise ValueError(f"A chromosome occurs in multiple splits: {path}")
    frame["split_local_index"] = frame.groupby("split", sort=False).cumcount()
    return frame


def bed_signature(frame):
    data = frame[BED_COLUMNS].to_csv(sep="\t", header=False, index=False).encode()
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_coordinates(handle, rows):
    handle.create_dataset(
        "chrom", data=rows.chrom.to_numpy(dtype=object), dtype=h5py.string_dtype()
    )
    for key in ("start", "end"):
        handle.create_dataset(key, data=rows[key].to_numpy(np.int64))


def validate_coordinates(handle, rows, strict=False):
    keys = {"chrom", "start", "end"}
    present = keys.intersection(handle.keys())
    if not present:
        if strict:
            raise ValueError(f"Missing coordinate metadata: {handle.filename}")
        return False
    if present != keys:
        raise ValueError(f"Incomplete coordinate metadata: {handle.filename}")
    matches = np.array_equal(handle["chrom"].asstr()[...], rows.chrom.to_numpy(str))
    for key in ("start", "end"):
        matches = matches and np.array_equal(handle[key][...], rows[key].to_numpy(np.int64))
    if not matches:
        raise ValueError(f"BED row order differs from HDF5 coordinates: {handle.filename}")
    return True


def embedding_path(directory, split):
    for name in (f"{split}_embeddings.h5", f"{split}_dna_embedding.h5"):
        path = Path(directory) / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No embedding H5 for split={split!r} under {directory}")
