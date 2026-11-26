# Experiment reproducibility

## Randomness and execution environment

Training seeds Python, NumPy, PyTorch, CUDA generators and DataLoader workers.
Signal selection uses a local deterministic RNG indexed by epoch and window.
Mixed-cell sampling uses its own recorded `sampler_seed` (historically 0),
independent from the model initialization seed (default 1401). Worker count,
sampler mode, batch size, signal file list, channel order and geometry all form
part of the experimental configuration.

The runtime explicitly seeds workers and data loaders. Its random number
streams and default device and batch settings can differ from the historical
DataParallel launchers. The full-input network operations and checkpoint
parameter names are preserved.

The runtime enables deterministic cuDNN selection and disables cuDNN
benchmarking and TF32. Set `strict_determinism=true` to require deterministic
algorithms. Operations without a deterministic implementation raise an error
in this mode. Hardware, software versions and batching can affect numerical
results. See the PyTorch documentation on
[deterministic algorithms](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html)
and [numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).

The smoke test trains a reduced model twice on CPU and compares all state-dict
tensors for exact equality. This test establishes repeatability for its
synthetic dataset and reference environment.

## Required experiment record

1. Release tag/commit and the installed wheel SHA-256.
2. Raw data accessions, assembly, reference FASTA checksum and index.
3. Accessibility processing tool versions and exact normalization/binning recipe.
4. ChIP target order and label-generation recipe, including missing-data rules.
5. Hi-C source URI, 10 kb resolution, raw/OE identity, ICE/balance recipe and masks.
6. Original split-labelled BED plus SHA-256; record exclusions/blacklists.
7. SEI checkpoint checksum and embedding generation precision.
8. Prepared input checksums. Set `hash_inputs=true` to record SHA-256 for all
   inputs. By default, CisLens records input sizes and modification times and
   hashes the BED and pretrained weights.
9. Full TOML, sampler seed, all selected cells, batch size and worker count.
10. `run.json`, `metrics.csv`, `train.log`, `best.pt` and its checksum.
11. `python -m pip freeze`, GPU model, NVIDIA driver, CUDA/cuDNN and environment.
12. Prediction command, split, output dtype and checkpoint hash.

Reproduction requires the original preprocessing, aggregation depth, split and
effective batch size. The legacy human-data preparation interpolated ChIP gaps,
clipped values and used float16 storage. The CisLens preparation command uses
masked, exact covered-base means and stores labels in float32. Use the archived
inputs for analyses based on the legacy preprocessing. Historical training
launchers used a batch size of 36, a learning rate of 0.0002, 80 epochs and a
seed of 1401 with multiple visible GPUs. Use the configuration recorded for the
specific experiment. The current trainer selects a single device.

## Checkpoint and output compatibility

The full-input network retains the original parameter names and operations.
Compatible legacy state dictionaries load strictly after removal of optional
`module.` prefixes. The loader uses the default architecture when a checkpoint
lacks architecture metadata. For checkpoints without target names, retain the
original target manifest and channel order.

Unsupported: RNA, ablation/depth-conditioned checkpoints, raw DNA training
without precomputed embeddings, arbitrary chromosome-grouped legacy embedding
formats, automatic ICE/OE, automatic multi-GPU training and exact optimizer/RNG
resume. Fine-tuning is initialization from pretrained model weights only.

Inference stores ChIP predictions and Hi-C predictions in the transformed
training space. The linear Hi-C decoder can produce negative values. Record
any inverse transformation, clipping or other postprocessing used in the
analysis.
