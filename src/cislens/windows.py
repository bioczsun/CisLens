"""Generate chromosome-held-out windows; existing paper BEDs should be reused."""

import numpy as np
import pandas as pd
from .fasta import IndexedFasta


def build_intervals(fasta_path, species, res, bins, stride_bins, seed, val_chr, test_chr):
    if min(res, bins, stride_bins) < 1:
        raise ValueError("res, bins and stride_bins must be positive")
    if val_chr == test_chr:
        raise ValueError("val_chr and test_chr must be different")
    if species not in {"human", "mouse"}:
        raise ValueError(f"Unsupported species: {species!r}")
    autosomes = 19 if species == "mouse" else 22
    allowed_chroms = {f"chr{index}" for index in range(1, autosomes + 1)} | {"chrX"}
    with IndexedFasta(fasta_path) as fasta:
        chr_lengths = {
            chrom: fasta.get_reference_length(chrom)
            for chrom in fasta.references
            if chrom in allowed_chroms
        }
    for chrom in (val_chr, test_chr):
        if chrom not in chr_lengths:
            raise ValueError(f"Split chromosome {chrom!r} is not a selected FASTA reference")
        if chr_lengths[chrom] < res * bins:
            raise ValueError(f"Split chromosome {chrom!r} is shorter than one sequence window")
    intervals = []
    for chrom, length in chr_lengths.items():
        split = "val" if chrom == val_chr else "test" if chrom == test_chr else "train"
        max_start_bin = length // res - bins
        for start_bin in range(0, max_start_bin + 1, stride_bins):
            start = start_bin * res
            intervals.append([chrom, start, start + bins * res, split])
    shuffle_indices = np.random.RandomState(seed).permutation(len(intervals))
    return pd.DataFrame(
        [intervals[index] for index in shuffle_indices], columns=["chrom", "start", "end", "split"]
    )
