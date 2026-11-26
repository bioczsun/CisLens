"""Reproducible FP32 training with explicit input provenance."""

from dataclasses import asdict
import csv
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

from .checkpoint import load_weights, read_checkpoint, save_checkpoint
from .data import build_datasets, make_loaders
from .losses import masked_supervision_loss, observed_window_count
from .metrics import PearsonMeter
from .runtime import environment, resolve_device, seed_everything, write_json
from .schema import sha256

LOGGER = logging.getLogger(__name__)


def evaluate(model, loader, device, channels):
    model.eval()
    totals = np.zeros(2)
    counts = np.zeros(2)
    pearson = PearsonMeter(channels)
    loss_functions = (torch.nn.PoissonNLLLoss(log_input=False), torch.nn.MSELoss())
    with torch.inference_mode():
        for batch in loader:
            signal, chip, hic, embedding = [value.to(device) for value in batch]
            outputs, prediction_hic = model(signal, embedding)
            pearson.update(outputs["epi"], chip)
            for index, (prediction, target, loss_fn) in enumerate(
                zip((outputs["epi"], prediction_hic), (chip, hic), loss_functions)
            ):
                count = observed_window_count(target)
                loss = masked_supervision_loss(prediction, target, loss_fn)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite validation loss")
                totals[index] += loss.item() * count
                counts[index] += count
    values = [
        float(total / count) if count else float("nan") for total, count in zip(totals, counts)
    ]
    return values[0], values[1], pearson.compute()


def input_manifest(groups, config):
    paths = set()
    for children in groups.values():
        for dataset in children:
            paths.update([dataset.supervision, dataset.embeddings, *dataset.signal_paths])
            # Legacy H5 coverage can be stored in a relative external link.
            with h5py.File(dataset.supervision, "r") as handle:
                for key in handle:
                    link = handle.get(key, getlink=True)
                    if isinstance(link, h5py.ExternalLink):
                        paths.add((dataset.supervision.parent / link.filename).resolve())
    for cell in config.cells or [""]:
        paths.add(Path(config.data_root) / cell / "sequences.bed")
    if config.pretrained:
        paths.add(Path(config.pretrained))
    result = []
    for path in sorted(paths):
        info = path.stat()
        entry = {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
        if config.hash_inputs or path.suffix == ".bed" or str(path) == config.pretrained:
            entry["sha256"] = sha256(path)
        result.append(entry)
    return result


def run(config, geometry, check_only=False):
    config.validate()
    seed_everything(config.seed, config.strict_determinism)
    groups, names = build_datasets(config, geometry)
    try:
        train_loader, val_loader = make_loaders(groups, config)
        LOGGER.info(
            "Validated %d train and %d val windows; targets=%s",
            len(train_loader.dataset),
            len(val_loader.dataset),
            names,
        )
        if check_only:
            return {
                "train": len(train_loader.dataset),
                "val": len(val_loader.dataset),
                "target_names": names,
            }
        device = resolve_device(config.device)
        model = geometry.build().to(device)
        if config.pretrained:
            payload = read_checkpoint(config.pretrained)
            load_weights(model, payload, names)
            names = names if names is not None else payload.get("target_names")
        output = Path(config.output_dir)
        # Refuse to mix runs or overwrite any previous experiment.
        output.mkdir(parents=True, exist_ok=False)
        handler = logging.FileHandler(output / "train.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
        try:
            write_json(
                output / "run.json",
                {
                    "training": asdict(config),
                    "model": asdict(geometry),
                    "environment": environment(),
                    "inputs": input_manifest(groups, config),
                    "target_names": names,
                },
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
            chip_loss = torch.nn.PoissonNLLLoss(log_input=False)
            hic_loss = torch.nn.MSELoss()
            best = float("inf")
            stale = 0
            with (output / "metrics.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "train_pearson",
                        "val_chip_loss",
                        "val_hic_loss",
                        "val_pearson",
                        "trained_batches",
                        "skipped_batches",
                    ]
                )
                for epoch in range(config.epochs):
                    for dataset in groups["train"]:
                        dataset.set_epoch(epoch)
                    if hasattr(train_loader.batch_sampler, "set_epoch"):
                        train_loader.batch_sampler.set_epoch(epoch)
                    model.train()
                    pearson = PearsonMeter(geometry.output_channels)
                    total = 0.0
                    trained = skipped = 0
                    for batch in train_loader:
                        if (
                            not torch.isfinite(batch[1]).any()
                            and not torch.isfinite(batch[2]).any()
                        ):
                            skipped += 1
                            continue
                        signal, chip, hic, embedding = [value.to(device) for value in batch]
                        optimizer.zero_grad(set_to_none=True)
                        epi, prediction_hic = model(signal, embedding)
                        loss = masked_supervision_loss(
                            prediction_hic, hic, hic_loss
                        ) + 0.01 * masked_supervision_loss(epi["epi"], chip, chip_loss)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                "Nonfinite training loss; inspect inputs and labels"
                            )
                        loss.backward()
                        optimizer.step()
                        pearson.update(epi["epi"], chip)
                        total += loss.item()
                        trained += 1
                    if not trained:
                        raise ValueError("Training epoch has no observed supervision")
                    val_chip, val_hic, val_pearson = evaluate(
                        model, val_loader, device, geometry.output_channels
                    )
                    stopping = val_hic if np.isfinite(val_hic) else val_chip
                    if not np.isfinite(stopping):
                        raise ValueError("Validation has no observed supervision")
                    writer.writerow(
                        [
                            epoch + 1,
                            total / trained,
                            pearson.compute(),
                            val_chip,
                            val_hic,
                            val_pearson,
                            trained,
                            skipped,
                        ]
                    )
                    handle.flush()
                    LOGGER.info(
                        "epoch=%d train=%.7g val_chip=%.7g val_hic=%.7g trained=%d skipped=%d",
                        epoch + 1,
                        total / trained,
                        val_chip,
                        val_hic,
                        trained,
                        skipped,
                    )
                    if stopping <= best:
                        best, stale = stopping, 0
                        save_checkpoint(
                            output / "best.pt",
                            model,
                            geometry,
                            config,
                            optimizer,
                            epoch + 1,
                            stopping,
                            names,
                        )
                    else:
                        stale += 1
                    if stale >= config.patience:
                        break
            return output / "best.pt"
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
    finally:
        for children in groups.values():
            for dataset in children:
                dataset.close()
