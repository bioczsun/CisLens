"""Build self-contained supervision H5 files from an explicit target manifest."""

from contextlib import ExitStack, closing
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .preparation_io import (
    HIC_BINS,
    HIC_RES,
    SEQ_RES,
    chromosome_alias,
    hic_window,
    observed_cool_bins,
    read_bigwig_bins,
    read_cool_band,
)
from .runtime import atomic_output
from .schema import (
    BED_COLUMNS,
    ModelConfig,
    SPLITS,
    TARGET_NAMES,
    bed_signature,
    read_bed,
    write_coordinates,
)


def run(bed, targets, output_dir, hic=None, hic_type="raw", balance=False, overwrite=False):
    import cooler
    import pyBigWig

    if hic_type not in {"raw", "oe"} or (hic_type == "oe" and balance):
        raise ValueError("hic_type must be raw/oe; do not balance an already normalized OE cool")
    geometry = ModelConfig()
    frame = read_bed(bed, geometry)
    manifest = Path(targets).resolve()
    table = pd.read_csv(manifest, sep="\t", dtype=str, keep_default_na=False)
    if (
        set(table.columns) != {"target", "path"}
        or table.target.duplicated().any()
        or not set(table.target).issubset(TARGET_NAMES)
    ):
        raise ValueError(
            "Targets TSV requires unique canonical target names and target/path columns"
        )
    sources = {
        row.target: (manifest.parent / row.path).resolve()
        for row in table.itertuples()
        if row.path.strip()
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Requested ChIP file does not exist: {path}; omit its row to mark it missing"
            )
    contact = cooler.Cooler(hic) if hic else None
    if contact is not None and contact.binsize != HIC_RES:
        raise ValueError(
            "Hi-C cool must have a 10000 bp resolution (specify the mcool group explicitly)"
        )
    if balance and contact is not None and "weight" not in contact.bins().columns:
        raise ValueError("balance requires precomputed cool weight values")
    output_dir = Path(output_dir)
    suffix = "_oe" if hic_type == "oe" else ""
    outputs = {split: output_dir / f"{split}_data{suffix}.h5" for split in SPLITS}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Supervision outputs already exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_bed = output_dir / "sequences.bed"
    normalized_bed = frame[BED_COLUMNS].to_csv(sep="\t", header=False, index=False)
    if output_bed.exists() and output_bed.read_text() != normalized_bed:
        raise ValueError(
            "Output directory already contains a different BED; use a separate directory"
        )
    if not output_bed.exists():
        with atomic_output(output_bed) as temporary:
            temporary.write_text(normalized_bed)
    observed_bins = observed_cool_bins(contact) if contact is not None else None
    with ExitStack() as publications:
        temporary = {
            split: publications.enter_context(atomic_output(path, overwrite))
            for split, path in outputs.items()
        }
        with ExitStack() as files:
            handles = {
                split: files.enter_context(h5py.File(path, "w"))
                for split, path in temporary.items()
            }
            bigwigs = {
                target: files.enter_context(closing(pyBigWig.open(str(path))))
                for target, path in sources.items()
            }
            for split, handle in handles.items():
                rows = frame[frame.split == split]
                write_coordinates(handle, rows)
                handle.attrs.update(
                    bed_sha256=bed_signature(frame),
                    hic_type=hic_type,
                    source_cool=str(hic or ""),
                    balance=balance,
                    missing_labels="NaN; mask with isfinite",
                    chip_summary="exact covered-base mean; no interpolation or clipping",
                )
                handle.create_dataset("target_names", data=TARGET_NAMES, dtype=h5py.string_dtype())
                handle.create_dataset(
                    "chip_available", data=[name in sources for name in TARGET_NAMES]
                )
                handle.create_dataset(
                    "source_files",
                    data=[str(sources.get(name, "")) for name in TARGET_NAMES],
                    dtype=h5py.string_dtype(),
                )
                for key, shape in (
                    ("coverage", (len(rows), geometry.seq_bins, len(TARGET_NAMES))),
                    ("hic", (len(rows), HIC_BINS, HIC_BINS)),
                ):
                    handle.create_dataset(
                        key,
                        shape=shape,
                        dtype="float32",
                        fillvalue=np.nan,
                        chunks=(1, *shape[1:]) if len(rows) else None,
                        compression="lzf" if len(rows) else None,
                    )
            for chrom, rows in frame.groupby("chrom", sort=False):
                left, right = int(rows.start.min()), int(rows.end.max())
                signals = np.full(
                    ((right - left) // SEQ_RES, len(TARGET_NAMES)), np.nan, dtype=np.float32
                )
                for index, name in enumerate(TARGET_NAMES):
                    if name in bigwigs:
                        signals[:, index] = read_bigwig_bins(
                            bigwigs[name], chrom, left, right, 65536
                        )
                band = starts = valid = None
                if contact is not None:
                    cool_chrom = chromosome_alias(chrom, contact.chromnames)
                    if cool_chrom is not None:
                        if right > contact.chromsizes[cool_chrom]:
                            raise ValueError(f"BED exceeds cool chromosome length: {chrom}")
                        starts, band, valid = read_cool_band(
                            contact, cool_chrom, balance, 4096, observed_bins
                        )
                for row in rows.itertuples(index=False):
                    offset = (row.start - left) // SEQ_RES
                    handles[row.split]["coverage"][row.split_local_index] = signals[
                        offset : offset + geometry.seq_bins
                    ]
                    if band is not None:
                        handles[row.split]["hic"][row.split_local_index] = hic_window(
                            starts, band, valid, row.start
                        )
    return outputs
