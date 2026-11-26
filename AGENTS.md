# Working on CisLens

This is a research implementation for ATAC/DNase training and inference.
Read README.md and docs/reproducibility.md before changing scientific behavior.

## Setup and checks

- Use Python 3.11 in an isolated environment and the reference constraints.
- Install the project; do not add src to sys.path in library code.
- Run `python -m pytest -q` and `cislens smoke --output-dir <new-directory>`.
- Run `cislens check --config <config.toml>` before a real training job.
- For changes to installation, build a wheel and test it outside the checkout.
- Keep CPU tests independent of internet access, pretrained weights and real data.

## Scientific invariants

- Retain full-input model parameter names for existing state-dict checkpoints.
- ATAC and DNase are the only supported input assays. Do not restore RNA,
  input ablations, depth experiments or downstream analyses without user intent.
- Default geometry: 2,097,152 bp, 1,024 SEI bins, 960 embedding channels,
  15 ChIP targets, and a 256×256 predicted Hi-C window.
- Raw Hi-C labels use ln(x+2), not ln(x+1). OE uses log2(1+x).
- Never independently reorder BED, embeddings, supervision or target channels.
- Missing labels remain NaN; recorded zeros remain observed.
- Preserve per-window loss normalization and the 0.01 ChIP loss weight.
- Do not silently ignore checkpoint shape/channel mismatches or reduce split
  validation to make an invalid dataset run.

## Experiment discipline

- Use explicit paths in TOML; they resolve relative to the config file.
- Real datasets/checkpoints are external inputs. Do not fetch large assets,
  launch production GPU jobs, delete data, or publish results without an
  applicable user instruction. Routine synthetic tests require no confirmation.
- Keep runs and predictions in new output paths. Preparation/inference only
  overwrite with an explicit user-selected option.
- Record configuration and provenance. Do not claim a test reproduces a
  manuscript result without its released datasets, weights and settings.
- A reduced CPU smoke model is a software test, not a benchmark.
- Describe the exact tests run and any GPU/installation checks not performed.
- Do not invent a DOI, accession, license, author list, or trained model URL.
