import numpy as np
import pytest
import torch

from cislens.checkpoint import load_weights, read_checkpoint
from cislens.data import transform_hic
from cislens.losses import masked_supervision_loss
from cislens.metrics import PearsonMeter
from cislens.runtime import atomic_output, seed_everything
from cislens.sampling import MixedCellBatchSampler
from cislens.schema import ModelConfig


@pytest.mark.parametrize("loss_fn", [torch.nn.MSELoss(), torch.nn.PoissonNLLLoss(log_input=False)])
def test_masked_loss_has_equal_window_weights_and_zero_missing_gradients(loss_fn):
    prediction = torch.tensor(
        [[[1.0, 1000.0], [4.0, 1000.0]], [[1.0, 2.0], [1000.0, 4.0]]], requires_grad=True
    )
    target = torch.tensor(
        [[[2.0, float("nan")], [2.0, float("nan")]], [[0.0, 3.0], [float("nan"), 3.0]]]
    )
    expected = torch.stack(
        [loss_fn(p[torch.isfinite(t)], t[torch.isfinite(t)]) for p, t in zip(prediction, target)]
    ).mean()
    result = masked_supervision_loss(prediction, target, loss_fn)
    torch.testing.assert_close(result, expected)
    result.backward()
    assert torch.all(prediction.grad[~torch.isfinite(target)] == 0)
    assert torch.isfinite(prediction.grad).all()
    missing = torch.ones((2, 3, 2), requires_grad=True)
    loss = masked_supervision_loss(missing, torch.full_like(missing, float("nan")), loss_fn)
    loss.backward()
    assert loss.item() == 0 and torch.all(missing.grad == 0)


def test_hic_transform_preserves_source_baseline_and_missingness():
    np.testing.assert_allclose(transform_hic(np.zeros((4, 4)), 8, "raw"), np.log(2))
    np.testing.assert_allclose(transform_hic(np.zeros((4, 4)), 8, "oe"), 0)
    assert np.isnan(transform_hic(np.full((4, 4), np.nan), 8, "raw")).all()
    partly_missing = np.ones((4, 4), dtype=np.float32)
    partly_missing[0] = np.nan
    transformed = transform_hic(partly_missing, 8, "raw")
    assert np.isfinite(transformed).any() and np.isnan(transformed).any()
    with pytest.raises(ValueError, match="nonnegative"):
        transform_hic(-np.ones((4, 4)), 8, "raw")


def test_pearson_ignores_missing_and_constant_channels():
    meter = PearsonMeter(3)
    targets = torch.tensor(
        [[[0.0, float("nan"), 1.0], [1.0, float("nan"), 1.0], [2.0, float("nan"), 1.0]]]
    )
    meter.update(torch.tensor([[[0.0, 99.0, 3.0], [1.0, 88.0, 3.0], [2.0, 77.0, 3.0]]]), targets)
    assert meter.compute() == pytest.approx(1)


def test_sampler_repeatability_epoch_changes_and_cell_balance():
    sampler = MixedCellBatchSampler([24, 24, 24], 6, seed=1401)
    first = list(sampler)
    assert first == list(sampler)
    for batch in first:
        assert [sum(index // 24 == cell for index in batch) for cell in range(3)] == [2, 2, 2]
    sampler.set_epoch(1)
    assert first != list(sampler)
    assert len(MixedCellBatchSampler([24, 24], 6, epoch_mode="fixed")) == 4
    with pytest.raises(ValueError, match="smaller"):
        MixedCellBatchSampler([24, 24, 24], 2)


def test_real_network_symmetry_gradient_and_legacy_weights(tmp_path):
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
    seed_everything(1401, strict=True)
    model = geometry.build()
    signal, embedding = torch.randn(2, 1, geometry.window), torch.randn(2, 32, 128)
    model.eval()
    epi, hic = model(signal, embedding)
    torch.testing.assert_close(hic, hic.transpose(1, 2))
    loss = epi["epi"].mean() + hic.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    checkpoint = tmp_path / "legacy.pth"
    torch.save(
        {"model_state_dict": {"module." + k: v for k, v in model.state_dict().items()}}, checkpoint
    )
    restored = geometry.build().eval()
    load_weights(restored, read_checkpoint(checkpoint))
    with torch.inference_mode():
        other_epi, other_hic = restored(signal, embedding)
    torch.testing.assert_close(epi["epi"], other_epi["epi"], rtol=0, atol=0)
    torch.testing.assert_close(hic, other_hic, rtol=0, atol=0)
    torch.save({"model_state_dict": {}, "input_mode": "sei_only"}, checkpoint)
    with pytest.raises(ValueError, match="ablation"):
        read_checkpoint(checkpoint)


def test_atomic_output_preserves_existing_data_on_failure(tmp_path):
    path = tmp_path / "result"
    path.write_text("complete")
    with pytest.raises(RuntimeError):
        with atomic_output(path, overwrite=True) as temporary:
            temporary.write_text("incomplete")
            raise RuntimeError("interrupted")
    assert path.read_text() == "complete"
    with pytest.raises(FileExistsError):
        with atomic_output(path):
            pass
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result"]
