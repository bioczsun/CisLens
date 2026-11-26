# Validation record

## CPU validation

Initial validation was completed on 2026-11-09 using synthetic inputs. The
resolved software environment and packaging results are recorded in
[tested_environment.json](tested_environment.json) and
[release_checks.json](release_checks.json).

- A Python 3.11 virtual environment installed the package, CPU PyTorch, and the
  preparation and development dependencies. `pip check` and
  `ruff check src tests` passed.
- All 22 tests passed, including repeated reduced-model ATAC-seq and DNase-seq
  training with exact parameter equality, checkpoint loading, and inference.
- Tests cover genomic window generation, missing labels, Cooler input,
  target order, BED and embedding alignment, legacy metadata compatibility,
  mixed-cell training and output integrity.
- The wheel and source distribution built successfully. Tests also passed
  against the wheel installed outside the checkout.

The 22 tests and the synthetic CPU workflow were repeated on 2026-11-26
against the installed CisLens 0.1.0 wheel in Python 3.11.2.

## CUDA validation

On 2026-11-26, CisLens 0.1.0 was tested on an NVIDIA A800 80 GB GPU with
PyTorch 2.5.1 and CUDA 12.1. A reduced model completed two training epochs
with synthetic ATAC-seq inputs. Checkpoint loading and inference produced
finite ChIP and Hi-C predictions.

The same implementation processed 32 held-out IMR-90 chromosome 15 windows
using the archived ATAC-seq checkpoint and corresponding inputs. The
predicted ChIP and Hi-C arrays matched the archived GPU outputs elementwise.
This comparison evaluated numerical consistency for the specified inference
subset.

## Model and checkpoint compatibility

A reduced full-input ChromEPI model and CisLens produced identical ChIP and
Hi-C predictions from the same weights and inputs. The original SEI embedding
branch and the adapted encoder also produced identical embeddings for a
2,048 bp sequence. The source SEI classifier was replaced with an identity
module for this comparison, which evaluated the convolutional embedding path.
See [parity_results.json](parity_results.json).

The legacy default-architecture DNase-seq checkpoint matched all 158 model
parameter names and tensor shapes. This check evaluated structural
compatibility.

## Validation scope

The tests establish installation and software behavior in the documented
environments, as well as inference consistency for the evaluated inputs.
Reproduction of manuscript results requires the corresponding data,
checkpoints, preprocessing, and analysis configuration. See
[experiment reproducibility](reproducibility.md).
