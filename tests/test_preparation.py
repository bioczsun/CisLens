import cooler
import h5py
import numpy as np
import pandas as pd
import pyBigWig
from pyfaidx import Fasta
import pytest
import torch

from cislens.embed import load_encoder, one_hot
from cislens.prepare import run as prepare
from cislens.schema import ModelConfig, TARGET_NAMES
from cislens.windows import build_intervals


def test_windows_have_stable_chromosome_splits_and_half_open_bounds(tmp_path):
    window = ModelConfig().window
    fasta = tmp_path / "reference.fa"
    with fasta.open("w") as handle:
        for chrom in ("chr1", "chr10", "chr15"):
            handle.write(f">{chrom}\n" + "A" * window + "\n")
    Fasta(str(fasta)).close()
    first = build_intervals(fasta, "human", 2048, 1024, 50, 1401, "chr10", "chr15")
    second = build_intervals(fasta, "human", 2048, 1024, 50, 1401, "chr10", "chr15")
    pd.testing.assert_frame_equal(first, second)
    assert dict(zip(first.chrom, first.split)) == {"chr1": "train", "chr10": "val", "chr15": "test"}
    assert (first.end == window).all() and (first.start == 0).all()


def test_supervision_exact_covered_mean_missing_channels_and_cool_grid(tmp_path):
    window = ModelConfig().window
    bed = tmp_path / "sequences.bed"
    bed.write_text(f"chr1\t0\t{window}\ttrain\nchr10\t0\t{window}\tval\nchr15\t0\t{window}\ttest\n")
    bw = tmp_path / "chip.bw"
    with pyBigWig.open(str(bw), "w") as handle:
        handle.addHeader([("chr1", window)], maxZooms=0)
        handle.addEntries(["chr1", "chr1"], [0, 2048], ends=[2048, 4096], values=[0.0, 3.0])
    manifest = tmp_path / "targets.tsv"
    manifest.write_text("target\tpath\nH3K27ac\tchip.bw\n")
    cool = tmp_path / "hic.cool"
    bins = cooler.binnify(pd.Series({"chr1": window}), 10000)
    positions = np.arange(len(bins))
    pixels = pd.DataFrame({"bin1_id": positions, "bin2_id": positions, "count": np.ones(len(bins))})
    cooler.create_cooler(str(cool), bins, pixels)
    output = tmp_path / "prepared"
    prepare(bed, manifest, output, hic=str(cool))
    with h5py.File(output / "train_data.h5", "r") as handle:
        assert tuple(handle["target_names"].asstr()[...]) == TARGET_NAMES
        assert handle["coverage"][0, 0, 0] == 0
        assert handle["coverage"][0, 1, 0] == 3
        assert np.isnan(handle["coverage"][0, 2:, :]).all()
        assert np.isnan(handle["coverage"][:, :, 1:]).all()
        assert handle["hic"].shape == (1, 209, 209)
        np.testing.assert_allclose(handle["hic"][0], np.eye(209))
    with h5py.File(output / "val_data.h5", "r") as handle:
        assert np.isnan(handle["hic"][...]).all()
        assert np.isnan(handle["coverage"][...]).all()
    with pytest.raises(FileExistsError):
        prepare(bed, manifest, output, hic=str(cool))


def test_sei_prefix_loading_and_unused_classifier_omission(tmp_path, monkeypatch):
    import cislens.embed

    class SmallEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lconv1 = torch.nn.Conv1d(4, 4, 1)

    model = SmallEncoder()
    path = tmp_path / "encoder.pt"
    state = {"module.model." + key: value for key, value in model.state_dict().items()}
    state["module.model.classifier.weight"] = torch.ones((1, 1))
    torch.save(state, path)
    monkeypatch.setattr(cislens.embed, "SeiEncoder", SmallEncoder)
    actual = load_encoder(path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value)
    assert np.array_equal(
        one_hot("ACGTNacgtn"), np.tile(np.concatenate([np.eye(4), np.zeros((1, 4))]), (2, 1))
    )
