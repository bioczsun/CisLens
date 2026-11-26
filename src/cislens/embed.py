"""SEI trunk embeddings; one immutable BED order for train/val/test."""

from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
import torch

from .runtime import atomic_output, resolve_device, seed_everything
from .checkpoint import load_payload
from .schema import ModelConfig, SPLITS, bed_signature, read_bed, sha256, write_coordinates
from .sei import SeiEncoder
from .fasta import IndexedFasta


def load_encoder(checkpoint):
    payload = load_payload(checkpoint)
    state = payload.get("state_dict", payload.get("model_state_dict", payload))
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.model.", "module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key.startswith(("classifier.", "spline_tr.")):
            continue
        cleaned[key] = value
    encoder = SeiEncoder()
    encoder.load_state_dict(cleaned, strict=True)
    return encoder


def one_hot(sequence):
    lookup = np.zeros((256, 4), dtype=np.float32)
    for i, bases in enumerate(("Aa", "Cc", "Gg", "Tt")):
        for base in bases:
            lookup[ord(base), i] = 1
    return lookup[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]


def run(
    bed,
    fasta,
    checkpoint,
    output_dir,
    device="auto",
    inner_batch=8,
    dtype="float16",
    overwrite=False,
):
    if inner_batch < 1 or dtype not in {"float16", "float32"}:
        raise ValueError("inner_batch must be positive; dtype must be float16/float32")
    geometry = ModelConfig()
    frame = read_bed(bed, geometry)
    output_dir = Path(output_dir)
    outputs = {split: output_dir / f"{split}_embeddings.h5" for split in SPLITS}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Embedding outputs already exist")
    if not Path(str(fasta) + ".fai").is_file():
        raise FileNotFoundError(
            "FASTA index is required; build it with pyfaidx or samtools faidx first"
        )
    seed_everything(1401)
    device = resolve_device(device)
    encoder = load_encoder(checkpoint).to(device).eval()
    signature = bed_signature(frame)
    checkpoint_hash = sha256(checkpoint)
    with IndexedFasta(fasta) as reference, ExitStack() as publications:
        temporary = {
            split: publications.enter_context(atomic_output(path, overwrite))
            for split, path in outputs.items()
        }
        with ExitStack() as files:
            handles = {
                split: files.enter_context(h5py.File(path, "w"))
                for split, path in temporary.items()
            }
            for split, handle in handles.items():
                rows = frame[frame.split == split]
                write_coordinates(handle, rows)
                shape = (len(rows), geometry.seq_bins, geometry.dim)
                handle.create_dataset(
                    "embeddings_tensor",
                    shape=shape,
                    dtype=dtype,
                    chunks=(1, geometry.seq_bins, geometry.dim) if len(rows) else None,
                )
                handle.attrs.update(
                    bed_sha256=signature,
                    sei_checkpoint_sha256=checkpoint_hash,
                    chunk_length=geometry.seq_res,
                    dtype=dtype,
                    amp=False,
                )
            with torch.inference_mode():
                for row in frame.itertuples(index=False):
                    if (
                        row.chrom not in reference.references
                        or row.end > reference.get_reference_length(row.chrom)
                    ):
                        raise ValueError(f"FASTA lacks interval {row.chrom}:{row.start}-{row.end}")
                    sequence = one_hot(reference.fetch(row.chrom, row.start, row.end))
                    chunks = sequence.reshape(geometry.seq_bins, geometry.seq_res, 4)
                    parts = []
                    for first in range(0, geometry.seq_bins, inner_batch):
                        batch = (
                            torch.from_numpy(chunks[first : first + inner_batch])
                            .permute(0, 2, 1)
                            .to(device)
                        )
                        parts.append(encoder(batch).cpu().numpy())
                    embedding = np.concatenate(parts).astype(dtype)
                    if not np.isfinite(embedding).all():
                        raise FloatingPointError(
                            "Nonfinite SEI embedding; use float32 or inspect weights"
                        )
                    handles[row.split]["embeddings_tensor"][row.split_local_index] = embedding
    return outputs
