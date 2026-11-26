"""Deterministic mixed-cell batches, preserving the original sampling policy."""

import torch
from torch.utils.data import Sampler


class MixedCellBatchSampler(Sampler):
    def __init__(self, dataset_lengths, batch_size, drop_last=True, seed=0, epoch_mode="concat"):
        if not dataset_lengths:
            raise ValueError("dataset_lengths must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size < len(dataset_lengths):
            raise ValueError(
                f"batch_size={batch_size} is smaller than number of cells={len(dataset_lengths)}. Increase batch size or reduce cells so each batch can include every cell."
            )
        self.dataset_lengths = [int(x) for x in dataset_lengths]
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch_mode = str(epoch_mode)
        self.epoch = 0
        if self.epoch_mode not in {"concat", "fixed"}:
            raise ValueError(
                f"Unsupported epoch_mode={epoch_mode!r}. Expected 'concat' or 'fixed'."
            )
        base = self.batch_size // len(self.dataset_lengths)
        remainder = self.batch_size % len(self.dataset_lengths)
        self.batch_sizes_per_cell = [
            base + (1 if idx < remainder else 0) for idx in range(len(self.dataset_lengths))
        ]
        self.offsets = []
        offset = 0
        for length in self.dataset_lengths:
            self.offsets.append(offset)
            offset += length

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self.epoch_mode == "concat":
            if self.drop_last:
                return min(
                    (
                        length // chunk
                        for length, chunk in zip(self.dataset_lengths, self.batch_sizes_per_cell)
                    )
                )
            return max(
                (
                    (length + chunk - 1) // chunk
                    for length, chunk in zip(self.dataset_lengths, self.batch_sizes_per_cell)
                )
            )
        reference_length = min(self.dataset_lengths)
        if self.drop_last:
            return reference_length // self.batch_size
        return (reference_length + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        per_cell_indices = []
        for length, offset in zip(self.dataset_lengths, self.offsets):
            local_indices = torch.randperm(length, generator=generator).tolist()
            per_cell_indices.append([offset + idx for idx in local_indices])
        positions = [0] * len(self.dataset_lengths)
        produced = 0
        target_batches = len(self)
        while produced < target_batches:
            batch = []
            for cell_idx, chunk_size in enumerate(self.batch_sizes_per_cell):
                start = positions[cell_idx]
                end = start + chunk_size
                chunk = per_cell_indices[cell_idx][start:end]
                if len(chunk) < chunk_size:
                    if self.drop_last:
                        return
                    continue
                batch.extend(chunk)
                positions[cell_idx] = end
            if len(batch) != self.batch_size:
                return
            batch = torch.tensor(batch)
            shuffle_order = torch.randperm(len(batch), generator=generator).tolist()
            yield batch[shuffle_order].tolist()
            produced += 1
