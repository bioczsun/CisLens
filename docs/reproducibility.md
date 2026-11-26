# Reproducing an experiment

## What is controlled

Training seeds Python, NumPy, PyTorch, CUDA generators and DataLoader workers.
Signal selection uses a local deterministic RNG indexed by epoch and window.
Mixed-cell sampling uses its own recorded `sampler_seed` (historically 0),
independent from the model initialization seed (default 1401). Worker count,
sampler mode, batch size, signal file list, channel order and geometry all form
part of the experimental configuration.

The restructured runtime supplies explicit worker/loader RNG management. Its
training RNG stream and default device/batch settings are not claimed to be
bitwise identical to historical monolithic/DataParallel launchers, even though
network operations and checkpoint parameter names are preserved.

cuDNN benchmarking and TF32 are disabled. cuDNN deterministic selection is
enabled. Strict deterministic algorithms are opt-in because unsupported CUDA
kernels raise rather than approximate. See the
[official PyTorch explanation](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html).
Different hardware/software/batching can produce different floating-point
results; see [numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
Do not promise cross-platform bitwise reproducibility from seeds alone.

The smoke workflow runs real reduced-model CPU training twice and checks every
state-dict tensor for exact equality. It verifies software determinism in that
environment. Production training uses much larger tensors; GPU determinism and
scientific accuracy require separate validation with the actual assets.

## Required experiment record

1. Release tag/commit and the installed wheel SHA-256.
2. Raw data accessions, assembly, reference FASTA checksum and index.
3. Accessibility processing tool versions and exact normalization/binning recipe.
4. ChIP target order and label-generation recipe, including missing-data rules.
5. Hi-C source URI, 10 kb resolution, raw/OE identity, ICE/balance recipe and masks.
6. Original split-labelled BED plus SHA-256; record exclusions/blacklists.
7. SEI checkpoint checksum and embedding generation precision.
8. Prepared input checksums. Set `hash_inputs=true` for the full training manifest;
   the default records sizes/mtime and only hashes BED/pretrained weights.
9. Full TOML, sampler seed, all selected cells, batch size and worker count.
10. `run.json`, `metrics.csv`, `train.log`, `best.pt` and its checksum.
11. `python -m pip freeze`, GPU model, NVIDIA driver, CUDA/cuDNN and environment.
12. Prediction command, split, output dtype and checkpoint hash.

Changing a label recipe, aggregation depth, split or batch size constitutes a
different experiment even if the network architecture is unchanged. The old
human builder interpolated gaps and clipped/stored float16 ChIP, while the new
optional builder uses masked exact covered-base means in float32. Reuse archived
inputs when comparing to earlier runs.

## Compatibility boundaries

The full-input network retains original names and operations. Compatible legacy
state dictionaries load strictly with optional `module.` prefixes removed.
Old checkpoint files lacking architecture metadata are interpreted as the default
architecture only. Old files lacking target names cannot establish semantic
channel identity; retain the original manifest and order.

Unsupported: RNA, ablation/depth-conditioned checkpoints, raw DNA training
without precomputed embeddings, arbitrary chromosome-grouped legacy embedding
formats, automatic ICE/OE, automatic multi-GPU training and exact optimizer/RNG
resume. Fine-tuning is initialization from pretrained model weights only.

Inference stores ChIP predictions and transformed-space Hi-C, not calibrated
counts. Hi-C outputs can be negative because its decoder is linear. Do not
exponentiate, clip or otherwise postprocess outputs without documenting it.
