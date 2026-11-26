# Validation record

Validation date: 2026-11-09. All numerical runs used synthetic inputs on CPU.
No production training job, biological-accuracy benchmark or data download was
performed. The only existing study asset inspected was a local DNase checkpoint,
read-only, for parameter-name/shape compatibility.

## Completed checks

- Full source inventory: 74 non-cache files; Python/notebook AST parsing and
  shell `bash -n`; per-file digests in source_manifest.json.
- A new Python 3.11 virtual environment, without system site packages, installed
  CPU PyTorch and the package with preparation/development dependencies.
  `pip check` reported no broken requirements. Exact resolved versions are in
  requirements/lock-linux-py311-cpu.txt and tested_environment.json.
- `ruff check src tests`: passed; code formatted with the pinned Ruff version.
- `python -m pytest -q`: **22 passed**, including real ATAC and DNase reduced-model
  training repeated with exact parameter equality, checkpoint reload and inference.
- Window generation, missing-label/zero handling, explicit cool preparation,
  ChIP channel semantics, BED/embedding row-order checks, old metadata fallback,
  all-missing training rejection, nested signal symlinks/cycle detection,
  real multi-cell training and atomic output preservation are covered.
- Wheel and source distributions build successfully. The installed wheel is
  separately checked outside the checkout before final delivery.

## Source numerical parity

See parity_results.json. A reduced full-input source ChromEPI network and the
reorganized CisLens network received identical weights and inputs. ChIP and
Hi-C maximum absolute differences were both **0.0**. The source SEI embedding
branch and retained encoder also differed by **0.0** on a 2,048 bp tensor.
For this test only, the unused source SEI Linear classifier was replaced with
Identity before construction to avoid allocating its multi-GB unused head;
the convolutional embedding path was otherwise unchanged.

The existing default-architecture DNase checkpoint matched all **158** network
state-tensor names and shapes. This is a structural compatibility check, not a
claim that full-size biological inference or the manuscript results were rerun.

## Limits

- No complete production-scale GPU training/inference or GPU memory benchmark.
- No validation of exact manuscript figures without released assets/configs.
- No external SEI checkpoint downloaded or full-genome embeddings regenerated.
- No claim that single-device training duplicates historical DataParallel runs.
- GitHub Actions is supplied as a CI workflow; it cannot be run on GitHub before
  the repository is actually published there.
- Cross-platform or cross-version bitwise equivalence is not guaranteed.

## Installation issues resolved during verification

Old pip failed canonical dependency-name parsing on the CPU PyTorch index.
The documented installer is now pinned. Old pysam releases required native
compilation on glibc 2.17; only indexed FASTA access was needed, so the optional
preparation dependency was replaced by pure-Python pyfaidx. Unbounded Pillow
selection likewise triggered native compilation; a compatible version is pinned.
The final clean environment installed successfully with these changes.
