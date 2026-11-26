import h5py
import pandas as pd
import pytest
import torch

from cislens.config import TrainConfig, load_config
from cislens.data import build_datasets
from cislens.schema import read_bed
from cislens.smoke import make_inputs


def inputs(tmp_path):
    geometry, signal = make_inputs(tmp_path / "data")
    config = TrainConfig(
        data_root=str(tmp_path / "data"),
        output_dir=str(tmp_path / "run"),
        device="cpu",
        batch_size=2,
        strict_metadata=True,
    )
    return geometry, config


def test_channel_order_conflict_detected_before_training(tmp_path):
    geometry, config = inputs(tmp_path)
    with h5py.File(tmp_path / "data/val_data.h5", "a") as handle:
        handle["target_names"][...] = ["CTCF", "H3K27ac"]
    with pytest.raises(ValueError, match="channel orders differ"):
        build_datasets(config, geometry)


def test_bed_coordinate_mismatch_detected(tmp_path):
    geometry, config = inputs(tmp_path)
    with h5py.File(tmp_path / "data/train_data.h5", "a") as handle:
        handle["start"][...] = [128, 0]
    with pytest.raises(ValueError, match="row order"):
        build_datasets(config, geometry)


def test_embedding_coordinate_mismatch_detected(tmp_path):
    geometry, config = inputs(tmp_path)
    with h5py.File(tmp_path / "data/dna_embedding/train_embeddings.h5", "a") as handle:
        handle["start"][...] = [128, 0]
    with pytest.raises(ValueError, match="row order"):
        build_datasets(config, geometry)


def test_legacy_metadata_compatibility_requires_no_new_fields(tmp_path):
    geometry, config = inputs(tmp_path)
    for path in (tmp_path / "data").rglob("*.h5"):
        with h5py.File(path, "a") as handle:
            for name in ("chrom", "start", "end", "target_names"):
                if name in handle:
                    del handle[name]
    config.strict_metadata = False
    groups, names = build_datasets(config, geometry)
    assert names is None
    for children in groups.values():
        for dataset in children:
            signal, chip, hic, embedding = dataset[0]
            assert signal.shape == (1, geometry.window)
            assert embedding.shape == (geometry.dim, geometry.seq_bins)
            assert torch.isfinite(hic).all()
            dataset.close()
    config.strict_metadata = True
    with pytest.raises(ValueError, match="coordinate metadata"):
        build_datasets(config, geometry)


def test_split_leakage_and_wrong_window_rejected(tmp_path):
    geometry, _ = inputs(tmp_path)
    path = tmp_path / "data/sequences.bed"
    frame = pd.read_csv(path, sep="\t", header=None)
    frame.loc[1, 0] = "chr1"
    frame.loc[1, 1:2] = [256, geometry.window + 256]
    frame.to_csv(path, sep="\t", header=False, index=False)
    with pytest.raises(ValueError, match="multiple splits"):
        read_bed(path, geometry)
    frame.loc[1, 2] += 1
    frame.to_csv(path, sep="\t", header=False, index=False)
    with pytest.raises(ValueError, match="windows must"):
        read_bed(path, geometry)


def test_config_paths_resolve_relative_to_config_and_unknown_fields_fail(tmp_path):
    path = tmp_path / "train.toml"
    path.write_text('[training]\ndata_root="data"\noutput_dir="run"\ndevice="cpu"\n')
    config, _ = load_config(path)
    assert config.data_root == str(tmp_path / "data")
    path.write_text('[training]\ndata_root="data"\noutput_dir="run"\nepoch_typo=1\n')
    with pytest.raises(TypeError, match="epoch_typo"):
        load_config(path)


def test_nested_signal_symlinks_are_loaded_and_cycles_are_skipped(tmp_path):
    geometry, config = inputs(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    signal_root = tmp_path / "data/atac"
    (signal_root / "synthetic.bigWig").rename(shared / "signal.bigWig")
    (signal_root / "50").symlink_to(shared, target_is_directory=True)
    (signal_root / "loop").symlink_to(signal_root, target_is_directory=True)
    groups, _ = build_datasets(config, geometry)
    for children in groups.values():
        for dataset in children:
            assert dataset.signal_counts == [50]
            assert torch.isfinite(dataset[0][0]).all()
            dataset.close()
