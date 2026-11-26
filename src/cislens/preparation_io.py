"""Bounded-memory supervision readers, preserving the source missing-bin policy."""

import numpy as np

SEQ_RES = 2048
HIC_RES = 10000
HIC_BINS = 209


def chromosome_alias(chrom, available):
    if chrom in available:
        return chrom
    alternate = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    return alternate if alternate in available else None


def read_bigwig_bins(handle, chrom, start, end, block_bins):
    values = np.full((end - start) // SEQ_RES, np.nan, dtype=np.float32)
    source_chrom = chromosome_alias(chrom, handle.chroms())
    if source_chrom is None:
        return values
    if end > handle.chroms(source_chrom):
        raise ValueError(f"BED exceeds bigWig chromosome length: {chrom}:{start}-{end}")
    for offset in range(0, len(values), block_bins):
        count = min(block_bins, len(values) - offset)
        left = start + offset * SEQ_RES
        block = np.asarray(
            handle.stats(
                source_chrom, left, left + count * SEQ_RES, nBins=count, type="mean", exact=True
            ),
            dtype=np.float64,
        )
        finite = np.isfinite(block)
        if np.any(block[finite] < 0) or np.any(block[finite] > np.finfo(np.float32).max):
            raise ValueError(f"Invalid nonnegative ChIP labels on {source_chrom}")
        block[~finite] = np.nan
        values[offset : offset + count] = block.astype(np.float32)
    return values


def observed_cool_bins(contact, block_pixels=1000000):
    observed = np.zeros(contact.info["nbins"], dtype=bool)
    pixels = contact.pixels()
    for first in range(0, contact.info["nnz"], block_pixels):
        block = pixels[first : first + block_pixels]
        counts = block["count"].to_numpy()
        positive = np.isfinite(counts) & (counts > 0)
        observed[block.loc[positive, "bin1_id"].to_numpy(dtype=np.int64)] = True
        observed[block.loc[positive, "bin2_id"].to_numpy(dtype=np.int64)] = True
    return observed


def read_cool_band(contact, chrom, balance, block_bins, observed_bins=None):
    """Read bounded sparse tiles; retain only the diagonals windows will use."""
    bins = contact.bins().fetch(chrom)
    starts = bins.start.to_numpy(dtype=np.int64)
    if np.any(starts % HIC_RES) or np.any(np.diff(starts) != HIC_RES):
        raise ValueError(f"Unexpected Hi-C bin grid: {chrom}")
    band = np.zeros((len(bins), 2 * HIC_BINS - 1), dtype=np.float32)
    chrom_offset = contact.offset(chrom)
    selector = contact.matrix(balance=balance, sparse=True)
    for first in range(0, len(bins), block_bins):
        last = min(first + block_bins, len(bins))
        col_first = max(0, first - HIC_BINS + 1)
        col_last = min(len(bins), last + HIC_BINS - 1)
        tile = selector[
            chrom_offset + first : chrom_offset + last,
            chrom_offset + col_first : chrom_offset + col_last,
        ].tocoo()
        tile.sum_duplicates()
        row_indices = tile.row + first
        offsets = tile.col + col_first - row_indices
        selected = np.abs(offsets) < HIC_BINS
        data = np.asarray(tile.data[selected], dtype=np.float32)
        if np.any(data[np.isfinite(data)] < 0):
            raise ValueError(f"Negative Hi-C labels: {chrom}")
        data[~np.isfinite(data)] = np.nan
        band[row_indices[selected], offsets[selected] + HIC_BINS - 1] = data
    if observed_bins is None:
        observed_bins = observed_cool_bins(contact)
    valid_bins = observed_bins[chrom_offset : chrom_offset + len(bins)].copy()
    if "weight" in bins:
        weights = bins.weight.to_numpy()
        valid_bins &= np.isfinite(weights) & (weights > 0)
    return (starts, band, valid_bins)


def hic_window(starts, band, valid_bins, start):
    coordinates = (int(start) // HIC_RES + np.arange(HIC_BINS)) * HIC_RES
    positions = np.searchsorted(starts, coordinates)
    valid = positions < len(starts)
    valid[valid] &= starts[positions[valid]] == coordinates[valid]
    valid[valid] &= valid_bins[positions[valid]]
    selected = positions[valid]
    result = np.full((HIC_BINS, HIC_BINS), np.nan, dtype=np.float32)
    offsets = selected[None, :] - selected[:, None] + HIC_BINS - 1
    result[np.ix_(valid, valid)] = band[selected[:, None], offsets]
    return result
