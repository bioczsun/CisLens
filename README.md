# CisLens

CisLens combines sequence-derived SEI embeddings with ATAC-seq or DNase-seq
accessibility signals to jointly predict epigenomic tracks and local Hi-C maps.
This repository contains training, inference, and the input preparation needed
by those workflows. RNA inputs, ablation experiments, loop calling, plotting,
and other downstream analyses are outside this release.

The code was reorganized from ChromEPI without changing its full-input network
computations or state-dict parameter names. See [the audit](docs/source_audit.md)
for file-by-file decisions and intentional engineering differences.

**Release status:** the synthetic CPU workflow is runnable without external
data or weights. Full-scale experiments require the reference genome, original
data splits, supervision, and SEI/CisLens weights. These assets and a manuscript
DOI have not been bundled or assigned download URLs. This code release alone
does not reproduce unpublished manuscript numbers. See
[publication requirements](docs/publication.md) and [validation](docs/validation.md).

## 1. Install

The reference environment is Linux x86-64 and Python 3.11. Python 3.12+ is
intentionally excluded because the reference NumPy/SciPy versions predate it.
Use a separate environment; do not replace packages in an existing research
environment. Commands below assume a downloaded/cloned repository named CisLens.

```bash
cd CisLens
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.3 setuptools==75.6.0 wheel==0.45.1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements/lock-linux-py311-cpu.txt -e ".[prepare,dev]"
python -m pip check
cislens --version
cislens doctor
```

For GPU work, replace both CPU PyTorch and package-install commands above with
the appropriate build and direct-constraint installation from
the [official PyTorch 2.5.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v251).
For example, CUDA 12.4 wheels use:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --prefer-binary -c requirements/constraints-py311.txt -e ".[prepare,dev]"
```

Install only the package for inference/training with already prepared inputs:
`python -m pip install -c requirements/lock-linux-py311-cpu.txt -e .` on CPU;
use the direct constraints instead for CUDA.
The `prepare` extra adds pyfaidx and cooler; the `dev` extra adds pytest and
packaging/lint tools. No optional analysis packages, Selene, SnapATAC2,
torchvision, or torchmetrics are needed.

`python -m cislens ...` is equivalent to `cislens ...` after installation.
`requirements/constraints-py311.txt` pins the direct numerical dependencies.
It is not a platform-independent lock file; retain `run.json` and the resolved
environment for each published experiment.

The default CPU commands use `requirements/lock-linux-py311-cpu.txt` **after**
installing CPU PyTorch. This file pins the resolved transitive dependencies as
well. Do not use the CPU lock
with a CUDA PyTorch build. The FASTA reader is pure Python (pyfaidx), avoiding
an unnecessary HTSlib build requirement. Pillow is pinned to a compatible wheel
instead of allowing future versions to trigger native compilation on old glibc.

## 2. Verify installation before using real data

```bash
python -m pytest -q
cislens smoke --output-dir smoke-output
```

The smoke command creates synthetic bigWig/H5 inputs, trains a **real reduced
CisLens network** twice on CPU, checks exact parameter equality, loads the best
checkpoint, and writes predictions. No pretrained weights or downloads are
required. It produces:

```text
smoke-output/
  data/
  run1/{best.pt,metrics.csv,run.json,train.log}
  run2/{best.pt,metrics.csv,run.json,train.log}
  prediction.h5
  smoke_result.json
```

Use a fresh output directory for every smoke run. Its reduced geometry and
synthetic signals test software behavior, not biological accuracy or production
memory requirements. The tests exercise both ATAC and DNase workflows.

## 3. Input contract

For multiple cells/tissues, arrange prepared data as:

```text
data/
  dna_embedding/
    train_embeddings.h5
    val_embeddings.h5
    test_embeddings.h5
  IMR90/
    sequences.bed
    train_data.h5
    val_data.h5
    test_data.h5
    atac/sample.bw
    dnase/sample.bigWig
  GM12878/
    sequences.bed
    train_data.h5
    val_data.h5
    test_data.h5
    atac/sample.bw
    dnase/sample.bigWig
```

Only the selected signal directory (`atac/` or `dnase/`) is required. One or
more `.bw`/`.bigWig` files may be stored recursively under it. For a single
cell, put `sequences.bed`, supervision, signal directories, and
`dna_embedding/` directly under `data_root`, and omit `cells` from the config.
Alternatively set `embedding_dir` to an explicit shared embedding directory.

BED is a **headerless, tab-separated, four-column** file:

```text
chr1	0	2097152	train
chr10	0	2097152	val
chr15	0	2097152	test
```

The separators above are actual tabs in real files. Coordinates are zero-based,
half-open. Default windows are 2,097,152 bp, comprising 1,024 bins of 2,048 bp.
Starts must align to 2,048 bp. A chromosome may appear in only one split.
Rows may be interleaved across splits; order **within each split** must match
its H5 files. All cells sharing DNA embeddings must use the same BED order and
split assignments. Never shuffle BED independently after preparing embeddings.
Do not mix genome assemblies or chromosome naming conventions across inputs.

| File / dataset | Default shape | Meaning |
| --- | --- | --- |
| `*_embeddings.h5:embeddings_tensor` | `[N_split,1024,960]` | SEI trunk embeddings; float16 or float32 |
| `*_data.h5:coverage` | `[N_split,1024,15]` | Nonnegative ChIP supervision |
| `*_data.h5:hic` | `[N_split,H,H]` | Nonnegative local Hi-C supervision, normally `H=209` |
| `chrom`, `start`, `end` | `[N_split]` | Optional legacy, required in strict metadata mode |
| `target_names` | `[15]` | ChIP output channel order |

The canonical channel order is:

```text
H3K27ac H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K36me3 H2AFZ H3K27me3
H3K9me3 H4K20me1 H3K79me2 CHD1 CTCF RAD21 SMC3
```

Missing labels remain **NaN**; recorded zero values remain supervised zeros.
Never fill missing labels using another tissue. Losses average valid elements
within a window and then average supervised windows, independently for ChIP
and Hi-C. Entirely unsupervised batches are skipped. An epoch with no
supervision, or a validation set with neither task observed, stops explicitly.

`strict_metadata = true` requires coordinates and channel names and is
recommended for newly generated data. Legacy root-level H5 datasets without
metadata are supported with warnings when false. Their unknown BED/channel
alignment cannot be proven automatically. Existing relative HDF5 external
links are supported, but linked files must move together. Newly prepared
CisLens supervision is self-contained.

## 4. Prepare inputs, if they are not already available

**To reproduce an existing experiment, reuse its exact BED, embeddings, and
labels.** Regenerating windows or changing missing-data processing changes the
experiment. In particular, the historical human preparation interpolated ChIP
gaps and stored float16; the new optional builder uses exact covered-base means
and NaN masking in float32. These are not interchangeable preprocessing recipes.

### 4.1 Reference genome and BED

Obtain the exact FASTA assembly used by the experiment and index it first:

```bash
python -c 'from pyfaidx import Fasta; Fasta("data/reference.fa").close()'
cislens windows --fasta data/reference.fa --species human \
  --output data/sequences.bed --seed 1401 --val-chr chr10 --test-chr chr15
```

`--species mouse` selects chr1–chr19 and chrX; human selects chr1–chr22 and chrX.
The default stride is 50 sequence bins (102,400 bp). chr10 is validation and
chr15 is test unless overridden. Other selected chromosomes are training.
No gap/blacklist filtering is applied automatically. A study-specific filtered
BED should be supplied directly and kept with the experiment.

### 4.2 SEI embeddings

Obtain the original SEI `sei.pth` through the
[upstream SEI distribution](https://github.com/FunctionLab/sei-framework#setup).
The upstream code/weights have separate terms; see
[third-party notices](THIRD_PARTY_NOTICES.md). No SEI weights are included here.

```bash
cislens embed --bed data/sequences.bed --fasta data/reference.fa \
  --checkpoint weights/sei.pth --output-dir data/dna_embedding \
  --device cuda:0 --inner-batch 8
```

Each 2,048 bp chunk is encoded as A/C/G/T one-hot; unknown bases are zero.
The 960-channel final convolutional representation is averaged over its spatial
axis. The unused SEI classification/spline head is omitted; encoder parameter
names and computations are preserved. Modern zip checkpoints are memory-mapped
while loading; older checkpoint formats may still need substantial host RAM.
Embedding extraction uses FP32 computation without AMP and defaults to float16
storage. Use `--dtype float32` if required by the original experiment.
The output includes coordinates, BED signature, and SEI checkpoint SHA-256.
Lower `--inner-batch` on memory errors; extraction time depends on hardware.

### 4.3 Accessibility bigWigs

Prepare the normalized **per-base accessibility signal** outside this package.
Use the same normalization and binning throughout a study. The source workflows
used CPM-normalized ATAC/DNase bigWigs; scATAC was aggregated to pseudobulk before
input. The model reads supplied values as-is, fills nonfinite input values with
zero, and does not apply an additional input log transform or CPM normalization.
Keep the upstream preprocessing command, tool version, and raw-data accessions.
ChIP is a target, not an additional input. RNA and per-cell expression inputs
are not accepted by this release.

### 4.4 ChIP and Hi-C supervision

Edit `examples/targets.tsv` for one cell. Paths resolve relative to the TSV;
rows identify canonical ChIP target names. Omit unavailable targets or use an
empty path to keep their channels masked. A specified nonexistent file is an
error, rather than silently being treated as missing.

```bash
cislens prepare --bed data/sequences.bed --targets examples/targets.tsv \
  --hic data/source_hic/IMR90.cool --hic-type raw \
  --output-dir data/IMR90
```

Input Hi-C must have a 10 kb grid. For mcool, specify its group explicitly:
`--hic 'data/source_hic/IMR90.mcool::/resolutions/10000'`.
The builder does not convert pairs, calculate ICE, or calculate OE. `--balance`
only applies existing weight values. For an already normalized OE cool, use
`--hic-type oe` without `--balance`; files are named `*_data_oe.h5` and training
must also use `hic_type = "oe"`. Omitting `--hic` keeps Hi-C labels missing and
allows ChIP-only supervision. Keep input assemblies consistent yourself.

ChIP uses exact means over recorded bases of each 2,048 bp bin, without gap
interpolation, clipping, or missing-to-zero conversion. Hi-C starts at
`floor(BED_start / 10000)` and extracts 209 bins. All-source positive contacts,
including long-range and trans contacts, determine covered bins. Uncovered bins
and invalid weights are masked; valid sparse zeros are retained. This is a
local prediction task, not a whole-genome contact reconstruction.

Repeat preparation for each cell using the same BED. Place that cell's ATAC or
DNase bigWigs under its selected signal directory. Existing outputs are refused
unless the specific preparation command receives `--overwrite`.

## 5. Train

Edit an example TOML. All config paths resolve relative to the **config file**;
explicit CLI path overrides resolve relative to the working directory.

```bash
cislens check --config examples/train_atac.toml
CUDA_VISIBLE_DEVICES=0 cislens train --config examples/train_atac.toml
```

For DNase:

```bash
cislens check --config examples/train_dnase.toml
CUDA_VISIBLE_DEVICES=0 cislens train --config examples/train_dnase.toml
```

`check` validates BED, H5 shapes, coordinates, known channel orders, signal
headers, and mixed-cell batching without training or creating run outputs.
It does not scan every label/embedding value; finite/nonnegative checks also
run when each window is loaded. Use it before scheduling a large run.

The example batch sizes are the minimum allowing one window per cell in a
mixed batch (3 for the ATAC example, 2 for DNase). These are portable examples,
**not claimed manuscript settings**. Historical launchers used batch=36,
lr=0.0002, epochs=80, seed=1401 and multiple visible GPUs. Reproducing such a run
requires its original effective batch size, device arrangement, data, and
configuration; this release uses one explicitly selected device, without
automatic DataParallel or DDP. Full production GPU memory/runtime has not been
benchmarked here. Reduce cells/batch or provision sufficient memory deliberately.

Important settings:

| Setting | Meaning |
| --- | --- |
| `signal_type` | `atac` or `dnase`; selects the corresponding directory |
| `hic_type` | `raw` → `*_data.h5`; `oe` → `*_data_oe.h5` |
| `batch_size` | Must be at least the number of jointly sampled cells |
| `seed` | Python, NumPy, PyTorch, and worker seeding; default 1401 |
| `sampler_seed` | Mixed-cell sampler seed; default 0 preserves the historical sampler |
| `epoch_mode` | `concat` or `fixed`; exact behavior below |
| `workers` | Default 0 for portability; workers keep independent handles |
| `strict_metadata` | Require BED coordinates and channel names |
| `strict_determinism` | Request deterministic algorithms, failing on unsupported operations |
| `hash_inputs` | Stream SHA-256 over every input; recommended for published runs |
| `pretrained` | Strict initialization from a compatible checkpoint; optional |

For mixed cells, every batch includes every cell. With `concat`, epoch length
is the minimum of `floor(N_cell / quota_cell)` and unused excess samples are
discarded for that epoch. This does **not** mean every concatenated sample is
seen. `fixed` uses `floor(min(N_cell) / batch_size)` batches. Samples and batch
order change deterministically with epoch. When using pseudobulk subdirectories
named 50/200/500/800/1000/1500/2000, signal sampling first chooses a supported
cell count uniformly, then a file within that group. Ungrouped files are ignored
when such groups exist, with a warning. Otherwise all discovered signals are
sampled uniformly. Validation always uses epoch zero.

Training uses FP32, AdamW (`weight_decay=0.01`), and:

```text
loss = MSE(Hi-C prediction, transformed Hi-C labels)
       + 0.01 * PoissonNLL(ChIP prediction, ChIP labels; log_input=False)
```

**Label transforms are preserved:** resize local Hi-C to 256×256, then use
`ln(x + 2)` for raw labels or `log2(1 + x)` for OE. Missing-aware resizing keeps
pixels masked unless their interpolation support is fully observed. Predictions
are written in this training space; Hi-C outputs are not raw contact counts and
are not automatically inverse-transformed. The ChIP head uses Softplus, while
the Hi-C decoder is linear and can produce negative values.

The historical Hi-C grid covers 209×10,000 bp from the floored start, whereas
the sequence BED window is 2,097,152 bp. The 256×256 image is a resized target
grid, not evidence of a new native experimental resolution. This convention is
retained for checkpoint compatibility.

Early stopping monitors validation Hi-C MSE, falling back to ChIP loss when
validation Hi-C is entirely missing. Patience defaults to 20. Metrics are
weighted by observed windows for each task. Undefined Pearson values may appear
as NaN in the CSV; a missing target is not converted to a negative example.

Each `output_dir` must be new. Outputs are `best.pt`, `metrics.csv`, `train.log`,
and `run.json`. The checkpoint contains architecture, channel names when known,
signal/Hi-C types, optimizer state, epoch, and training configuration. The run
record includes environment and input file metadata; BED and pretrained weight
hashes are always recorded, and `hash_inputs=true` hashes large inputs too.
`pretrained` initializes model weights only; exact mid-run optimizer/RNG resume
is not implemented. Do not describe initialization as resuming an experiment.

## 6. Infer

Inference requires only a compatible checkpoint, one signal bigWig, the original
split-labelled BED, and matching DNA embeddings. ChIP/Hi-C labels are unnecessary.
Choose a checkpoint trained with the intended input assay and normalization.

```bash
cislens infer --checkpoint runs/atac_seed1401/best.pt \
  --signal data/IMR90/atac/sample.bw --bed data/IMR90/sequences.bed \
  --embedding-dir data/dna_embedding --split test --device cuda:0 \
  --batch-size 4 --output results/IMR90_test.h5
```

For DNase, supply the DNase-trained checkpoint and DNase bigWig. Use `--split all`
for all BED rows in their original order, `--device cpu` for CPU, and
`--strict-metadata` for coordinate validation of new embeddings. Default storage
is float32; `--dtype float16` is optional and rejects overflow. Inference does
not use AMP. Output files are published only after successful completion, and
existing predictions require explicit `--overwrite` to replace.

Default output datasets:

```text
pred_epi            [N,1024,15]
pred_hic            [N,256,256]
chrom/start/end     [N]
split               [N]
split_local_index   [N]
target_names        [15] when present in the checkpoint
```

The H5 also records checkpoint/BED hashes, environment, input signal, and model
metadata. No loops, figures, `.cool` export, or analysis steps run automatically.

Legacy full-input ChromEPI `model_state_dict` checkpoints with the default
architecture are accepted, including `module.` prefixes. Missing metadata
cannot be inferred reliably. Ablation and depth-conditioned experimental weights
are explicitly outside this release. Loading is strict; incompatible shapes
are not silently skipped. Checkpoints are loaded using PyTorch's restricted
`weights_only=True` reader; do not disable this globally to suppress an error.

## 7. Reproducibility and agent use

For a paper, archive the exact BED order, input processing recipe, file hashes,
checkpoint hash, config, `run.json`, metrics, software environment, GPU model,
driver/CUDA versions, batch size, and train/validation/test assignments.
Use immutable data/weight download records and a release tag. Full guidance is
in [reproducibility](docs/reproducibility.md).

Seeds, deterministic cuDNN selection, and disabled TF32 reduce variability.
CPU smoke runs verify repeated parameter equality in the tested environment.
They do not establish bitwise identity across different devices, library
versions, batch sizes, or distributed strategies. `strict_determinism=true`
can reject CUDA operations with no deterministic implementation, consistent
with the [PyTorch determinism documentation](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html).

For Codex and other coding agents, start with [AGENTS.md](AGENTS.md): install in
an isolated environment, run smoke/tests, inspect config, run `check`, then train
or infer with a new output path. Never fix schema mismatches by disabling
validation, shuffling files independently, or inventing missing observations.

## 8. Repository and licensing

```text
src/cislens/       installable runtime, CLI, data preparation and model
tests/             synthetic numerical, I/O and workflow regression tests
examples/          explicit ATAC/DNase TOML configurations and target manifest
requirements/      reference dependency constraints
docs/              audit, data/reproduction notes and validation evidence
.github/workflows/ CPU CI template
LICENSES/          retained third-party license text
```

Original CisLens author/license/citation information is pending. No general
open-source license has been chosen on the authors' behalf. The SEI-derived
encoder retains its upstream academic/research-use license in
[LICENSES/SEI.txt](LICENSES/SEI.txt); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Complete [the publication metadata](docs/publication.md) before sharing the
repository as an officially released manuscript implementation.
