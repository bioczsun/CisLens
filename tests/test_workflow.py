import h5py
import numpy as np
import pytest
import shutil

from cislens.config import TrainConfig
from cislens.infer import run as infer
from cislens.smoke import make_inputs, run as smoke
from cislens.train import run as train


@pytest.mark.parametrize("signal_type", ["atac", "dnase"])
def test_repeated_real_training_and_inference(tmp_path, signal_type):
    result = smoke(tmp_path / signal_type, signal_type)
    assert result["status"] == "passed"
    assert result["repeated_parameters_equal"]
    with h5py.File(tmp_path / signal_type / "prediction.h5", "r") as handle:
        assert tuple(handle["target_names"].asstr()[...]) == ("H3K27ac", "CTCF")
        assert handle.attrs["signal_type"] == signal_type


def test_check_only_writes_nothing_and_missing_supervision_is_rejected(tmp_path):
    geometry, _ = make_inputs(tmp_path / "data")
    config = TrainConfig(
        data_root=str(tmp_path / "data"),
        output_dir=str(tmp_path / "run"),
        device="cpu",
        batch_size=2,
        epochs=1,
    )
    assert train(config, geometry, check_only=True)["train"] == 2
    assert not (tmp_path / "run").exists()
    with h5py.File(tmp_path / "data/train_data.h5", "a") as handle:
        handle["coverage"][...] = np.nan
        handle["hic"][...] = np.nan
    with pytest.raises(ValueError, match="no observed supervision"):
        train(config, geometry)


def test_inference_failure_does_not_replace_complete_output(tmp_path):
    smoke(tmp_path / "smoke")
    directory = tmp_path / "smoke"
    output = directory / "prediction.h5"
    original = output.read_bytes()
    with h5py.File(directory / "data/dna_embedding/test_embeddings.h5", "a") as handle:
        handle["embeddings_tensor"][...] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        infer(
            directory / "run1/best.pt",
            directory / "data/atac/synthetic.bigWig",
            directory / "data/sequences.bed",
            directory / "data/dna_embedding",
            output,
            device="cpu",
            overwrite=True,
        )
    assert output.read_bytes() == original


def test_real_multicell_training_with_shared_embeddings(tmp_path):
    geometry, _ = make_inputs(tmp_path / "data")
    root = tmp_path / "data"
    for cell in ("first", "second"):
        directory = root / cell
        directory.mkdir()
        for name in ("sequences.bed", "train_data.h5", "val_data.h5", "test_data.h5"):
            shutil.copyfile(root / name, directory / name)
        (directory / "atac").symlink_to(root / "atac", target_is_directory=True)
    config = TrainConfig(
        data_root=str(root),
        output_dir=str(tmp_path / "run"),
        cells=["first", "second"],
        device="cpu",
        batch_size=2,
        epochs=1,
        strict_metadata=True,
    )
    assert train(config, geometry, check_only=True)["train"] == 4
    checkpoint = train(config, geometry)
    assert checkpoint.is_file()
    import pandas as pd

    assert pd.read_csv(tmp_path / "run/metrics.csv").trained_batches.iloc[0] == 2
