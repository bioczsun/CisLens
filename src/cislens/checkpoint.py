"""Portable, strict state-dict loading with full-input legacy compatibility."""

from dataclasses import asdict
import zipfile

import torch

from .runtime import atomic_output
from .schema import ModelConfig


def load_payload(path):
    # Modern checkpoints can map large classifier/optimizer tensors lazily.
    return torch.load(path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path))


def read_checkpoint(path):
    payload = load_payload(path)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Checkpoint must contain model_state_dict")
    if payload.get("input_mode", "full") != "full":
        raise ValueError(
            "This release supports full-input ATAC/DNase checkpoints; ablation checkpoints are excluded"
        )
    state = {
        key.removeprefix("module."): value for key, value in payload["model_state_dict"].items()
    }
    if any(key.startswith("depth_mlp.") for key in state):
        raise ValueError("Depth-conditioned experimental checkpoints are outside this release")
    payload = dict(payload)
    payload["model_state_dict"] = state
    return payload


def load_weights(model, payload, target_names=None):
    names = payload.get("target_names")
    if target_names is not None and names is not None and tuple(names) != tuple(target_names):
        raise ValueError("Checkpoint ChIP channel order differs from training data")
    model.load_state_dict(payload["model_state_dict"], strict=True)


def load_model(path, device):
    payload = read_checkpoint(path)
    geometry = ModelConfig(**payload.get("model_config", {}))
    model = geometry.build()
    load_weights(model, payload)
    names = payload.get("target_names")
    if names is not None and len(names) != geometry.output_channels:
        raise ValueError("Checkpoint target_names do not match model output channels")
    return model.to(device).eval(), geometry, payload


def save_checkpoint(path, model, geometry, config, optimizer, epoch, val_loss, target_names):
    payload = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(geometry),
        "training_config": asdict(config),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": float(val_loss),
        "input_mode": "full",
        "signal_type": config.signal_type,
        "hic_type": config.hic_type,
        "target_names": target_names,
    }
    with atomic_output(path, overwrite=True) as temporary:
        torch.save(payload, temporary)
