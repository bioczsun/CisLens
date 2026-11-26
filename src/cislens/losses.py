"""Masked supervision losses shared by training and validation."""

import torch


def masked_supervision_loss(predictions, targets, loss_fn):
    """Average valid labels per window, then average windows with supervision."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} differs from target shape {targets.shape}"
        )
    losses = []
    for prediction, target in zip(predictions, targets):
        valid = torch.isfinite(target)
        if valid.any():
            losses.append(loss_fn(prediction[valid], target[valid]).mean())
    if not losses:
        return predictions.reshape(-1)[:0].sum()
    return torch.stack(losses).mean()


def observed_window_count(targets):
    return int(torch.isfinite(targets).flatten(1).any(dim=1).sum().item())
