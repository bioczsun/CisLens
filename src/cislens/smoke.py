"""A real, reduced CPU training/inference cycle; no pretrained weights required."""

from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyBigWig
import torch

from .config import TrainConfig
from .infer import run as infer
from .runtime import write_json
from .schema import BED_COLUMNS, ModelConfig, SPLITS, write_coordinates
from .train import run as train


def make_inputs(directory, signal_type="atac"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    geometry = ModelConfig(
        dim=32,
        qk_dim=4,
        v_dim=4,
        num_head=2,
        depth=2,
        mid_dim=8,
        seq_bins=128,
        seq_res=128,
        output_channels=2,
    )
    window = geometry.window
    frame = pd.DataFrame(
        [
            ["chr1", 0, window, "train"],
            ["chr10", 0, window, "val"],
            ["chr1", 128, window + 128, "train"],
            ["chr15", 0, window, "test"],
        ],
        columns=BED_COLUMNS,
    )
    frame.to_csv(directory / "sequences.bed", sep="\t", header=False, index=False)
    embedding_dir = directory / "dna_embedding"
    embedding_dir.mkdir()
    rng = np.random.default_rng(1401)
    for split in SPLITS:
        rows = frame[frame.split == split]
        with h5py.File(directory / f"{split}_data.h5", "w") as handle:
            write_coordinates(handle, rows)
            handle.create_dataset(
                "target_names", data=("H3K27ac", "CTCF"), dtype=h5py.string_dtype()
            )
            coverage = np.abs(rng.normal(size=(len(rows), geometry.seq_bins, 2))).astype(np.float32)
            coverage[:, ::3, 1] = np.nan
            handle.create_dataset("coverage", data=coverage)
            handle.create_dataset("hic", data=np.ones((len(rows), 16, 16), dtype=np.float32))
            handle.attrs["hic_type"] = "raw"
        with h5py.File(embedding_dir / f"{split}_embeddings.h5", "w") as handle:
            write_coordinates(handle, rows)
            handle.create_dataset(
                "embeddings_tensor",
                data=rng.normal(size=(len(rows), geometry.seq_bins, geometry.dim)).astype(
                    np.float32
                ),
            )
    signal_dir = directory / signal_type
    signal_dir.mkdir()
    signal = signal_dir / "synthetic.bigWig"
    with pyBigWig.open(str(signal), "w") as handle:
        handle.addHeader([("chr1", window + 128), ("chr10", window), ("chr15", window)], maxZooms=0)
        handle.addEntries(
            ["chr1", "chr10", "chr15"],
            [0, 0, 0],
            ends=[window + 128, window, window],
            values=[2.0, 3.0, 4.0],
        )
    return geometry, signal


def run(output_dir, signal_type="atac"):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        geometry, signal = make_inputs(output / "data", signal_type)
        config = TrainConfig(
            data_root=str(output / "data"),
            output_dir=str(output / "run1"),
            signal_type=signal_type,
            device="cpu",
            batch_size=2,
            epochs=2,
            strict_metadata=True,
            strict_determinism=True,
            hash_inputs=True,
        )
        first = train(config, geometry)
        config.output_dir = str(output / "run2")
        second = train(config, geometry)
        one = torch.load(first, map_location="cpu", weights_only=True)["model_state_dict"]
        two = torch.load(second, map_location="cpu", weights_only=True)["model_state_dict"]
        if any(not torch.equal(one[key], two[key]) for key in one):
            raise AssertionError("Repeated seeded CPU training produced different parameters")
        predictions = output / "prediction.h5"
        infer(
            first,
            signal,
            output / "data/sequences.bed",
            output / "data/dna_embedding",
            predictions,
            device="cpu",
            strict_metadata=True,
        )
        with h5py.File(predictions, "r") as handle:
            if handle["pred_epi"].shape != (1, 128, 2) or handle["pred_hic"].shape != (1, 32, 32):
                raise AssertionError("Unexpected smoke prediction shape")
            if (
                not np.isfinite(handle["pred_epi"][...]).all()
                or not np.isfinite(handle["pred_hic"][...]).all()
            ):
                raise AssertionError("Smoke predictions must be finite")
        result = {
            "status": "passed",
            "signal_type": signal_type,
            "repeated_parameters_equal": True,
            "model": asdict(geometry),
            "prediction": "prediction.h5",
            "checkpoint": "run1/best.pt",
        }
        write_json(output / "smoke_result.json", result)
        return result
    finally:
        torch.set_num_threads(previous_threads)
