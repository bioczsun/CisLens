"""Strict TOML configuration; data paths resolve relative to the config file."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .schema import ModelConfig


@dataclass
class TrainConfig:
    data_root: str = ""
    output_dir: str = ""
    cells: list[str] | None = None
    embedding_dir: str | None = None
    pretrained: str | None = None
    signal_type: str = "atac"
    hic_type: str = "raw"
    batch_size: int = 4
    epochs: int = 80
    learning_rate: float = 0.0002
    weight_decay: float = 0.01
    patience: int = 20
    workers: int = 0
    seed: int = 1401
    sampler_seed: int = 0
    epoch_mode: str = "concat"
    device: str = "cuda"
    strict_determinism: bool = False
    strict_metadata: bool = False
    hash_inputs: bool = False

    def validate(self):
        if not self.data_root or not self.output_dir:
            raise ValueError("training.data_root and training.output_dir are required")
        if self.signal_type not in {"atac", "dnase"} or self.hic_type not in {"raw", "oe"}:
            raise ValueError("signal_type must be atac/dnase; hic_type must be raw/oe")
        if min(self.batch_size, self.epochs, self.patience) < 1 or self.workers < 0:
            raise ValueError(
                "batch_size, epochs and patience must be positive; workers must be nonnegative"
            )
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.seed < 0 or self.seed >= 2**32:
            raise ValueError("Invalid optimizer settings or seed")
        if self.epoch_mode not in {"concat", "fixed"}:
            raise ValueError("epoch_mode must be concat or fixed")
        if self.cells is not None and (
            not isinstance(self.cells, list)
            or any(
                not isinstance(c, str) or not c or "/" in c or "\\" in c or c in {".", ".."}
                for c in self.cells
            )
            or len(self.cells) != len(set(self.cells))
        ):
            raise ValueError("cells must contain unique directory names")


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    if set(document) - {"training", "model"}:
        raise ValueError("Configuration only accepts [training] and [model] sections")
    training = TrainConfig(**document.get("training", {}))
    geometry = ModelConfig(**document.get("model", {}))
    for key in ("data_root", "output_dir", "embedding_dir", "pretrained"):
        value = getattr(training, key)
        if value:
            value = Path(value).expanduser()
            setattr(training, key, str((path.parent / value).resolve()))
    training.validate()
    return training, geometry
