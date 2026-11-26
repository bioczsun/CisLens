# CisLens

CisLens integrates sequence-derived SEI embeddings with chromatin accessibility
profiles to predict epigenomic tracks and local Hi-C contact maps. This
repository provides data preparation, model training and inference for ATAC-seq
and DNase-seq inputs.

The synthetic workflow requires no external datasets or checkpoints.
Study-specific workflows require the reference assembly, original data split,
prepared inputs and appropriate model weights. See
[software and data availability](docs/publication.md) and
[validation](docs/validation.md).

## 1. Installation

The reference environment is Linux x86-64 with Python 3.11. Run the following
commands from a checkout named `CisLens` to install the package in a virtual
environment.

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

For GPU execution, use the PyTorch build compatible with the installed NVIDIA
driver and install CisLens with the direct dependency constraints. The commands
below use CUDA 12.4. See the
[PyTorch 2.5.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v251).

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --prefer-binary -c requirements/constraints-py311.txt -e ".[prepare,dev]"
```

Training and inference with prepared inputs require the base package. The
`prepare` extra installs pyfaidx and Cooler for input preparation. The `dev`
extra installs testing, linting and build tools.

`python -m cislens ...` is equivalent to `cislens ...` after installation. The
CPU lock file specifies the complete reference CPU environment. CUDA installations
use the direct constraints in `requirements/constraints-py311.txt`. Record the
resolved environment and `run.json` with each experiment.

## 2. Installation verification

```bash
python -m pytest -q
cislens smoke --output-dir smoke-output
```

The smoke test creates synthetic bigWig and HDF5 inputs, trains a reduced
CisLens model twice on CPU, and verifies exact equality of the resulting model
parameters. It then reloads a checkpoint and writes predictions. The output
directory contains the following files.

```text
smoke-output/
  data/
  run1/{best.pt,metrics.csv,run.json,train.log}
  run2/{best.pt,metrics.csv,run.json,train.log}
  prediction.h5
  smoke_result.json
```

Specify a new output directory for each run. The test suite covers ATAC-seq and
DNase-seq training and inference with synthetic inputs. See the
[validation record](docs/validation.md) for the scope of the tests.

## 3. Input data

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
Rows may be interleaved across splits. Within each split, the BED row order
must match the corresponding HDF5 datasets. Cells that share DNA embeddings
require identical BED row order and split assignments. All inputs must use a
consistent genome assembly and chromosome naming convention.

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

Missing labels are encoded as NaN. Observed zeros contribute to the loss.
Losses are averaged over valid elements within each window and then over
supervised windows, independently for ChIP and Hi-C. Batches without supervision
are skipped. Training requires at least one supervised batch per epoch and
observed labels for at least one task in the validation set.

`strict_metadata = true` requires coordinates and channel names and is
recommended for newly generated data. Legacy root-level H5 datasets without
metadata are supported with warnings when false. Their unknown BED/channel
alignment cannot be proven automatically. Existing relative HDF5 external
links are supported, but linked files must move together. Newly prepared
CisLens supervision is self-contained.

## 4. Data preparation

Reproduction requires the original BED, embeddings and labels. The archived
human-data workflow interpolated gaps in ChIP coverage and stored labels in
float16. The preparation command below uses exact covered-base means, NaN
masking and float32 storage. Use the original input files for analyses based
on the archived workflow.

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

Supply normalized per-base accessibility signals as bigWig files. The study
workflows used counts-per-million normalization for ATAC-seq and DNase-seq and
aggregated single-cell ATAC-seq to pseudobulk. The model uses the supplied signal
scale and replaces nonfinite input values with zero. Record the source data
accessions, preprocessing commands, software versions and normalization
settings. ChIP tracks provide supervision during training.

### 4.4 ChIP and Hi-C supervision

Edit `examples/targets.tsv` for the cell of interest. The rows identify
canonical ChIP targets, and paths resolve relative to the TSV. Omit unavailable
targets or leave their paths empty to mask their channels. Each specified path
must reference an existing file.

```bash
cislens prepare --bed data/sequences.bed --targets examples/targets.tsv \
  --hic data/source_hic/IMR90.cool --hic-type raw \
  --output-dir data/IMR90
```

Input Hi-C must have a 10 kb grid. For mcool, specify its group explicitly:
`--hic 'data/source_hic/IMR90.mcool::/resolutions/10000'`.
The builder does not convert pairs, calculate ICE, or calculate OE. `--balance`
only applies existing weight values. For an already normalized OE cool, use
`--hic-type oe` and leave `--balance` unset. The resulting files are named
`*_data_oe.h5` and require `hic_type = "oe"` at training time. Omit `--hic` for
ChIP-only supervision. The source files and BED must use the same genome assembly.

ChIP uses exact means over recorded bases of each 2,048 bp bin, without gap
interpolation, clipping, or missing-to-zero conversion. Hi-C starts at
`floor(BED_start / 10000)` and extracts 209 bins. All-source positive contacts,
including long-range and trans contacts, determine covered bins. Uncovered bins
and invalid weights are masked. Valid sparse zeros are retained. Each target
matrix describes contacts within a local genomic window.

Repeat preparation for each cell using the same BED. Place that cell's ATAC or
DNase bigWigs under its selected signal directory. Existing outputs are refused
unless the specific preparation command receives `--overwrite`.

## 5. Training

Edit an example TOML. All config paths resolve relative to the **config file**;
explicit CLI path overrides resolve relative to the working directory.

```bash
cislens check --config examples/train_atac.toml
CUDA_VISIBLE_DEVICES=0 cislens train --config examples/train_atac.toml
```

The DNase-seq workflow uses its corresponding configuration.

```bash
cislens check --config examples/train_dnase.toml
CUDA_VISIBLE_DEVICES=0 cislens train --config examples/train_dnase.toml
```

`check` validates BED intervals, HDF5 shapes, available coordinate and channel
metadata, signal headers, and mixed-cell batching. Device initialization,
checkpoint loading and value-level checks occur during training.

The example configurations allocate at least one window per cell in each mixed
batch. Reproduction of an archived experiment requires its recorded input data,
configuration, effective batch size and device arrangement. CisLens training
uses a single selected device.

### Training options

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

For mixed cells, every batch includes every cell. With `concat`, the number of
batches per epoch is the minimum of `floor(N_cell / quota_cell)` across cells.
Samples exceeding the corresponding per-cell allocation are omitted for that
epoch. `fixed` uses `floor(min(N_cell) / batch_size)` batches. Samples and batch
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

Local Hi-C maps are resized to 256×256 before applying
`ln(x + 2)` for raw labels or `log2(1 + x)` for OE. Missing-aware resizing keeps
pixels masked unless their interpolation support is fully observed. Hi-C
predictions are stored in the transformed training space. The ChIP head uses
Softplus. The linear Hi-C decoder can produce negative values.

The native Hi-C grid comprises 209 bins at 10 kb resolution, beginning at the
bin containing the BED start. The sequence window spans 2,097,152 bp. The
256×256 prediction grid is defined by resizing the native Hi-C grid.

Early stopping monitors validation Hi-C MSE, falling back to ChIP loss when
validation Hi-C is entirely missing. Patience defaults to 20. Metrics are
weighted by observed windows for each task. Undefined Pearson values may appear
as NaN in the CSV; a missing target is not converted to a negative example.

Each `output_dir` must be new. Outputs are `best.pt`, `metrics.csv`, `train.log`,
and `run.json`. The checkpoint contains architecture, channel names when known,
signal/Hi-C types, optimizer state, epoch, and training configuration. The run
record includes environment and input file metadata; BED and pretrained weight
hashes are always recorded, and `hash_inputs=true` hashes large inputs too.
`pretrained` initializes model parameters for a new run. Optimizer and random
number generator states are initialized separately.

## 6. Inference

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

The output HDF5 contains the following datasets.

```text
pred_epi            [N,1024,15]
pred_hic            [N,256,256]
chrom/start/end     [N]
split               [N]
split_local_index   [N]
target_names        [15] when present in the checkpoint
```

The HDF5 output also records checkpoint and BED hashes, the software
environment, the input signal and model metadata.

The loader accepts legacy full-input ChromEPI `model_state_dict` checkpoints
for the default architecture, including keys prefixed with `module.`.
Parameter loading requires matching tensor names and shapes and uses PyTorch's
restricted `weights_only=True` reader. Keep the associated input and target
manifests when using legacy checkpoints.

## 7. Reproducibility

Archive the exact BED and split assignments, input preparation settings, file
and checkpoint hashes, configuration, `run.json`, metrics and software and
hardware versions. Record the repository revision with each experiment. See
[reproducibility](docs/reproducibility.md) for the complete experiment record.

## 8. Repository and licensing

```text
src/cislens/       installable runtime, CLI, data preparation and model
tests/             synthetic numerical, I/O and workflow regression tests
examples/          explicit ATAC/DNase TOML configurations and target manifest
requirements/      reference dependency constraints
docs/              software availability, reproducibility and validation
.github/workflows/ continuous integration
LICENSES/          third-party license terms
```

Model compatibility and implementation provenance are documented in
[source provenance](docs/source_audit.md). The SEI-derived encoder retains
its upstream academic and research-use terms. See
[the SEI license](LICENSES/SEI.txt) and
[third-party notices](THIRD_PARTY_NOTICES.md).
