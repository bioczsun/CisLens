"""Window inference without labels; completed outputs are published atomically."""

from pathlib import Path

import h5py
import numpy as np
import pyBigWig
import torch
from torch.utils.data import DataLoader, Dataset

from .checkpoint import load_model
from .runtime import atomic_output, environment, resolve_device, seed_everything
from .schema import (
    bed_signature,
    embedding_path,
    read_bed,
    sha256,
    validate_coordinates,
    write_coordinates,
)


class PredictionDataset(Dataset):
    def __init__(self, rows, full_bed, signal, embedding_dir, geometry, strict=False):
        self.rows = rows.reset_index(drop=True)
        self.signal = Path(signal)
        self.embedding_dir = Path(embedding_dir)
        self.geometry = geometry
        self._bw = None
        self._files = {}
        with pyBigWig.open(str(signal)) as handle:
            sizes = handle.chroms()
            for chrom, group in rows.groupby("chrom", sort=False):
                if chrom not in sizes or group.end.max() > sizes[chrom]:
                    raise ValueError(f"Signal lacks BED intervals for {chrom}: {signal}")
        for split in rows.split.unique():
            path = embedding_path(embedding_dir, split)
            expected_rows = full_bed[full_bed.split == split]
            with h5py.File(path, "r") as handle:
                expected = (len(expected_rows), geometry.seq_bins, geometry.dim)
                if handle["embeddings_tensor"].shape != expected:
                    raise ValueError(f"Embedding shape must be {expected}: {path}")
                validate_coordinates(handle, expected_rows, strict)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        if self._bw is None:
            self._bw = pyBigWig.open(str(self.signal))
        if row.split not in self._files:
            self._files[row.split] = h5py.File(embedding_path(self.embedding_dir, row.split), "r")
        signal = self._bw.values(row.chrom, int(row.start), int(row.end), numpy=True)
        signal = np.nan_to_num(signal, nan=0, posinf=0, neginf=0).astype(np.float32)
        embedding = self._files[row.split]["embeddings_tensor"][int(row.split_local_index)].astype(
            np.float32
        )
        if not np.isfinite(embedding).all():
            raise ValueError("DNA embeddings contain nonfinite values")
        return torch.from_numpy(signal[None, :]), torch.from_numpy(embedding.T.copy())

    def close(self):
        if self._bw is not None:
            self._bw.close()
        for handle in self._files.values():
            handle.close()
        self._bw = None
        self._files = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_bw=None, _files={})
        return state


def run(
    checkpoint,
    signal,
    bed,
    embedding_dir,
    output,
    split="test",
    batch_size=4,
    workers=0,
    device="auto",
    dtype="float32",
    overwrite=False,
    strict_metadata=False,
):
    if batch_size < 1 or workers < 0 or dtype not in {"float16", "float32"}:
        raise ValueError("Invalid inference batch size, worker count or dtype")
    if Path(output).exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}")
    seed_everything(1401)
    device = resolve_device(device)
    model, geometry, payload = load_model(checkpoint, device)
    frame = read_bed(bed, geometry)
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split must be train/val/test/all")
    rows = frame if split == "all" else frame[frame.split == split]
    if rows.empty:
        raise ValueError(f"No BED windows selected for split={split}")
    dataset = PredictionDataset(rows, frame, signal, embedding_dir, geometry, strict_metadata)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    try:
        with atomic_output(output, overwrite) as temporary, h5py.File(temporary, "w") as handle:
            write_coordinates(handle, rows)
            handle.create_dataset(
                "split", data=rows.split.to_numpy(dtype=object), dtype=h5py.string_dtype()
            )
            handle.create_dataset(
                "split_local_index", data=rows.split_local_index.to_numpy(np.int64)
            )
            names = payload.get("target_names")
            if names is not None:
                handle.create_dataset("target_names", data=names, dtype=h5py.string_dtype())
            epi = handle.create_dataset(
                "pred_epi",
                shape=(len(rows), geometry.seq_bins, geometry.output_channels),
                dtype=dtype,
                chunks=(1, geometry.seq_bins, geometry.output_channels),
            )
            image_size = geometry.seq_bins // geometry.pool_size
            hic = handle.create_dataset(
                "pred_hic",
                shape=(len(rows), image_size, image_size),
                dtype=dtype,
                chunks=(1, image_size, image_size),
            )
            offset = 0
            with torch.inference_mode():
                for input_signal, embedding in loader:
                    outputs, prediction_hic = model(input_signal.to(device), embedding.to(device))
                    predictions = (outputs["epi"].cpu().numpy(), prediction_hic.cpu().numpy())
                    if any(not np.isfinite(v).all() for v in predictions):
                        raise FloatingPointError("Nonfinite model prediction")
                    count = len(predictions[0])
                    for target, prediction in zip((epi, hic), predictions):
                        converted = prediction.astype(dtype)
                        if not np.isfinite(converted).all():
                            raise FloatingPointError("Prediction overflow; use float32 output")
                        target[offset : offset + count] = converted
                    offset += count
            handle.attrs.update(
                checkpoint_sha256=sha256(checkpoint),
                bed_sha256=bed_signature(frame),
                signal=str(Path(signal).resolve()),
                input_mode="full",
                signal_type=payload.get("signal_type", "unspecified_legacy"),
                hic_type=payload.get("hic_type", "unspecified_legacy"),
                seed=1401,
                output_dtype=dtype,
            )
            import json

            handle.attrs["environment"] = json.dumps(environment())
    finally:
        dataset.close()
    return Path(output)
