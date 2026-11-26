"""Finite-label Pearson correlation; no downstream analysis dependencies."""

import numpy as np


class PearsonMeter:
    def __init__(self, channels):
        self.sums = np.zeros((6, channels), dtype=np.float64)

    def update(self, predictions, targets):
        x = predictions.detach().cpu().numpy().astype(np.float64)
        y = targets.detach().cpu().numpy().astype(np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        x = np.where(valid, x, 0)
        y = np.where(valid, y, 0)
        self.sums += np.stack(
            [
                valid.sum((0, 1)),
                x.sum((0, 1)),
                y.sum((0, 1)),
                (x * x).sum((0, 1)),
                (y * y).sum((0, 1)),
                (x * y).sum((0, 1)),
            ]
        )

    def compute(self):
        n, x, y, xx, yy, xy = self.sums
        with np.errstate(divide="ignore", invalid="ignore"):
            values = (xy - x * y / n) / np.sqrt(
                np.maximum(xx - x * x / n, 0) * np.maximum(yy - y * y / n, 0)
            )
        valid = values[np.isfinite(values)]
        return float(valid.mean()) if len(valid) else float("nan")
