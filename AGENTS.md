# Development guidelines

Review [the README](README.md) and
[reproducibility documentation](docs/reproducibility.md) before modifying the
model, data preparation or training workflow.

## Setup and checks

- Use Python 3.11 in an isolated environment with the reference constraints.
- Install CisLens as a package and use its installed import paths.
- Run `python -m pytest -q` and `cislens smoke --output-dir <new-directory>`.
- Run `cislens check --config <config.toml>` before a real training job.
- Verify installation changes by testing the built wheel outside the checkout.
- Use synthetic inputs for CPU tests so they can run offline.

## Scientific invariants

- Retain full-input model parameter names for existing state-dict checkpoints.
- The supported input assays are ATAC-seq and DNase-seq. Keep changes to
  other workflows within the scope requested by the user.
- Default geometry: 2,097,152 bp, 1,024 SEI bins, 960 embedding channels,
  15 ChIP targets, and a 256×256 predicted Hi-C window.
- Transform raw Hi-C labels with ln(x+2) and OE labels with log2(1+x).
- Keep BED rows, embeddings, supervision and target channels aligned.
- Missing labels remain NaN; recorded zeros remain observed.
- Preserve per-window loss normalization and the 0.01 ChIP loss weight.
- Preserve strict checkpoint validation and validation of the data splits.

## Experiment provenance and execution

- Specify input paths in TOML relative to the configuration file or as
  absolute paths.
- Download large assets, run production GPU jobs, delete data, or publish
  results only with specific authorization. Routine synthetic tests may run
  offline without additional approval.
- Write runs and predictions to new output paths. Overwriting preparation
  and inference outputs requires an explicit option.
- Record the configuration, inputs and validation scope. Manuscript
  reproduction requires the corresponding datasets, weights and settings.
- Report the tests performed and the environment used. Record synthetic tests
  separately from performance and accuracy benchmarks.
- Verify citations, accessions, license terms, authorship and download links
  against authoritative records before inclusion.
