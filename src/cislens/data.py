"""Validated training inputs and worker-local bigWig/HDF5 handles."""

from collections import OrderedDict
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pyBigWig
from skimage.transform import resize
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .runtime import seed_worker
from .sampling import MixedCellBatchSampler
from .schema import bed_signature, embedding_path, read_bed, validate_coordinates

LOGGER = logging.getLogger(__name__)


def discover_signal_paths(root):
    paths = []
    seen = set()
    for directory, children, files in os.walk(root, followlinks=True):
        children.sort()
        canonical = os.path.realpath(directory)
        if canonical in seen:
            children[:] = []
            LOGGER.warning("Skipping repeated/cyclic signal directory: %s", directory)
            continue
        seen.add(canonical)
        for filename in files:
            path = Path(directory) / filename
            if path.is_file() and path.suffix.lower() in {".bw", ".bigwig"}:
                paths.append(path)
    return sorted(paths, key=str)


def transform_hic(values, image_size, hic_type):
    valid = np.isfinite(values)
    if np.any(values[valid] < 0):
        raise ValueError("Hi-C supervision must be nonnegative before the log transform")
    shape = (image_size, image_size)
    if valid.all():
        matrix = resize(values, shape, anti_aliasing=True)
    else:
        numerator = resize(np.where(valid, values, 0), shape, anti_aliasing=True)
        weights = resize(valid.astype(np.float32), shape, anti_aliasing=True)
        matrix = np.full(shape, np.nan, dtype=np.float32)
        observed = weights >= 1 - 1e-6
        matrix[observed] = numerator[observed] / weights[observed]
    if hic_type == "raw":
        return np.log1p(matrix + 1).astype(np.float32)
    if hic_type == "oe":
        return np.log2(1 + matrix).astype(np.float32)
    raise ValueError(f"Unsupported hic_type: {hic_type}")


class WindowDataset(Dataset):
    SIGNAL_CELL_COUNTS = (50, 200, 500, 800, 1000, 1500, 2000)

    def __init__(
        self, rows, supervision, embeddings, signals, geometry, hic_type="raw", strict=False
    ):
        self.rows = rows.reset_index(drop=True)
        self.supervision = Path(supervision)
        self.embeddings = Path(embeddings)
        self.geometry = geometry
        self.hic_type = hic_type
        self.epoch = 0
        self._h5 = self._emb = None
        self._bw = OrderedDict()
        self.signal_paths = discover_signal_paths(signals)
        if not self.signal_paths:
            raise FileNotFoundError(f"No .bw/.bigWig signal files under {signals}")
        grouped = {}
        for path in self.signal_paths:
            parents = path.relative_to(signals).parts[:-1]
            count = next(
                (
                    int(v)
                    for v in reversed(parents)
                    if v.isdigit() and int(v) in self.SIGNAL_CELL_COUNTS
                ),
                None,
            )
            if count is not None:
                grouped.setdefault(count, []).append(path)
        self.grouped_signals = grouped
        self.signal_counts = sorted(grouped)
        used = sum(map(len, grouped.values()))
        if grouped and used != len(self.signal_paths):
            LOGGER.warning(
                "Ignoring %d ungrouped signals while sampling by cell count: %s",
                len(self.signal_paths) - used,
                signals,
            )
        required_ends = self.rows.groupby("chrom").end.max()
        active = [p for group in grouped.values() for p in group] if grouped else self.signal_paths
        for path in active:
            with pyBigWig.open(str(path)) as handle:
                sizes = handle.chroms()
                for chrom, end in required_ends.items():
                    if chrom not in sizes or end > sizes[chrom]:
                        raise ValueError(f"Signal lacks BED intervals for {chrom}: {path}")
        with h5py.File(self.supervision, "r") as handle:
            expected = (len(rows), geometry.seq_bins, geometry.output_channels)
            if handle["coverage"].shape != expected:
                raise ValueError(
                    f"Output head size or ChIP coverage shape mismatch: expected {expected}, got {handle['coverage'].shape}: {supervision}"
                )
            hic = handle["hic"]
            if hic.ndim != 3 or hic.shape[0] != len(rows) or hic.shape[1] != hic.shape[2]:
                raise ValueError(f"Invalid Hi-C supervision shape: {supervision}")
            if handle.attrs.get("hic_type", hic_type) != hic_type:
                raise ValueError(f"Hi-C type differs from config: {supervision}")
            self.target_names = (
                tuple(handle["target_names"].asstr()[...]) if "target_names" in handle else None
            )
            if self.target_names is not None and (
                len(self.target_names) != geometry.output_channels
                or len(set(self.target_names)) != len(self.target_names)
            ):
                raise ValueError(f"Invalid target_names in {supervision}")
            has_coords = validate_coordinates(handle, self.rows, strict)
        with h5py.File(self.embeddings, "r") as handle:
            expected = (len(rows), geometry.seq_bins, geometry.dim)
            if handle["embeddings_tensor"].shape != expected:
                raise ValueError(f"Embedding shape must be {expected}: {embeddings}")
            emb_coords = validate_coordinates(handle, self.rows, strict)
        if not has_coords or not emb_coords or self.target_names is None:
            if strict:
                raise ValueError(
                    f"Strict metadata mode requires coordinates and target_names: {supervision}"
                )
            LOGGER.warning(
                "Legacy metadata: only shapes/available coordinates can be checked for %s; keep the original BED and channel order",
                supervision,
            )

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        if self._h5 is None:
            self._h5 = h5py.File(self.supervision, "r")
            self._emb = h5py.File(self.embeddings, "r")
        rng = np.random.RandomState((self.epoch * max(len(self), 1) + index) % (2**32))
        if self.signal_counts:
            choices = self.grouped_signals[int(rng.choice(self.signal_counts))]
        else:
            choices = self.signal_paths
        path = choices[int(rng.randint(len(choices)))]
        if path not in self._bw:
            self._bw[path] = pyBigWig.open(str(path))
        self._bw.move_to_end(path)
        if len(self._bw) > 16:
            self._bw.popitem(last=False)[1].close()
        handle = self._bw[path]
        chroms = handle.chroms()
        if row.chrom not in chroms or row.end > chroms[row.chrom]:
            raise ValueError(f"Signal lacks BED interval {row.chrom}:{row.start}-{row.end}: {path}")
        signal = handle.values(row.chrom, int(row.start), int(row.end), numpy=True)
        signal = np.nan_to_num(signal, nan=0, posinf=0, neginf=0).astype(np.float32)
        labels = self._h5["coverage"][index].astype(np.float32)
        if np.any(labels[np.isfinite(labels)] < 0):
            raise ValueError(f"ChIP supervision must be nonnegative: {self.supervision}")
        hic = transform_hic(
            self._h5["hic"][index], self.geometry.seq_bins // self.geometry.pool_size, self.hic_type
        )
        embedding = self._emb["embeddings_tensor"][index].astype(np.float32)
        if not np.isfinite(embedding).all():
            raise ValueError(f"Nonfinite DNA embedding: {self.embeddings}, row {index}")
        return tuple(
            torch.from_numpy(v) for v in (signal[None, :], labels, hic, embedding.T.copy())
        )

    def close(self):
        for handle in (self._h5, self._emb, *self._bw.values()):
            if handle is not None:
                handle.close()
        self._h5 = self._emb = None
        self._bw.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_h5=None, _emb=None, _bw=OrderedDict())
        return state


def validate_dataset_targets(datasets):
    names = None
    for dataset in datasets:
        if dataset.target_names is not None:
            if names is not None and dataset.target_names != names:
                raise ValueError(f"ChIP channel orders differ: {dataset.supervision}")
            names = dataset.target_names
    return names


def build_datasets(config, geometry):
    root = Path(config.data_root)
    embedding_dir = Path(config.embedding_dir) if config.embedding_dir else root / "dna_embedding"
    groups = {split: [] for split in ("train", "val")}
    signature = None
    for cell in config.cells or [""]:
        directory = root / cell
        bed = read_bed(directory / "sequences.bed", geometry)
        current = bed_signature(bed)
        if signature is not None and signature != current:
            raise ValueError(
                "Cells sharing DNA embeddings must use identical BED row orders and splits"
            )
        signature = current
        for split, datasets in groups.items():
            rows = bed[bed.split == split]
            if rows.empty:
                raise ValueError(f"No {split} windows for {directory}")
            suffix = "_oe" if config.hic_type == "oe" else ""
            datasets.append(
                WindowDataset(
                    rows,
                    directory / f"{split}_data{suffix}.h5",
                    embedding_path(embedding_dir, split),
                    directory / config.signal_type,
                    geometry,
                    config.hic_type,
                    config.strict_metadata,
                )
            )
    names = validate_dataset_targets(groups["train"] + groups["val"])
    return groups, names


def make_loaders(groups, config):
    def loader(split):
        children = groups[split]
        dataset = children[0] if len(children) == 1 else ConcatDataset(children)
        options = dict(
            num_workers=config.workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config.seed),
            pin_memory=config.device.startswith("cuda"),
        )
        if split == "train" and len(children) > 1:
            sampler = MixedCellBatchSampler(
                [len(d) for d in children],
                config.batch_size,
                seed=config.sampler_seed,
                epoch_mode=config.epoch_mode,
            )
            if not len(sampler):
                raise ValueError("Batch size leaves zero mixed-cell batches; reduce batch_size")
            return DataLoader(dataset, batch_sampler=sampler, **options)
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, **options)

    return loader("train"), loader("val")
