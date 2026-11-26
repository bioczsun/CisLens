# Implementation provenance and compatibility

The source review covers 74 ChromEPI source, configuration and notebook files.
The [source manifest](source_manifest.json) records paths, SHA-256 checksums,
top-level definitions, machine-specific path references and syntax results.

Python modules and notebook code cells passed syntax parsing, and shell
scripts passed `bash -n`. The active training, network, dataset, checkpoint
and data preparation components received numerical and interface review.
Review of archived components was limited to static analysis.

## Implementation conventions

1. Machine-specific default data/genome/interpreter/checkpoint paths were removed.
   All input locations are explicit CLI/TOML values.
2. Multiple archived architectures/monolithic trainers are not part of the active
   inference path; they are excluded instead of being renamed into public APIs.
3. Full-input network operations and state-dict parameter names are retained.
   Numerical parity against the source model is checked separately.
4. SEI's embedding return precedes its large classifier/spline head. Keep only
   encoder modules and ignore only known unused-head checkpoint prefixes.
5. Source raw Hi-C preprocessing uses ln(x+2), and OE uses log2(1+x). Preserve
   these explicit transforms rather than silently correcting their unusual baseline.
6. The human and mouse preparation recipes differ (float16/interpolation/clipping
   versus masked float32 covered-base means). They are documented as different
   experiments; archived inputs are needed to reproduce earlier runs.
7. Source channel validation is meaningful: same-shaped channels can have different
   biological identities. Keep strict checks for known names and old-H5 compatibility.
8. Embedding shape alone cannot prove alignment. New embeddings carry coordinates
   and a canonical BED signature; strict mode rejects missing coordinates.
9. Silent partial checkpoint loading is replaced by strict state-dict loading.
   Old full-input/default-architecture checkpoints remain supported; experimental
   input/depth variants are explicitly rejected.
10. Sampling policy is preserved, including the historical mixed sampler seed 0.
    `concat` stops at the minimum per-cell quota rather than exhaustively consuming
    all concatenated samples; this is documented without changing the algorithm.
11. Workers are configurable instead of fixed at ten; HDF5/bigWig handles are
    worker-local and explicitly closed. Preflight checks signal chromosome bounds.
12. Missing observations stay NaN; finite zeros are retained. Empty batches do not
    update model/optimizer; validation weights each task by observed windows.
13. The runtime reports diagnostic errors for nonfinite losses and predictions.
14. Run reuse is refused. Checkpoints/predictions are staged and atomically
    published; a failed prediction does not overwrite an existing complete output.
15. Reproducibility records include seeds, model configuration, software and
    hardware details, input metadata and optional full input checksums.
16. The supported runtime covers ATAC-seq and DNase-seq input preparation,
    training and inference. RNA inputs, ablation and depth-conditioned models,
    and downstream export and analysis are documented as historical components.
17. Historical training launchers used DataParallel when multiple GPUs were
    visible. The current trainer uses one selected device. Record the device
    configuration and effective batch size for each experiment.
18. The SEI-derived encoder retains the academic and research-use terms of the
    [upstream license](../LICENSES/SEI.txt).
19. The reference environment pins installation and build tools, uses pyfaidx
    for indexed FASTA access and specifies a compatible Pillow wheel for the
    tested Linux environment.

## Exact duplicate files

- `model copy/blocks copy.py`; `model copy/blocks.py`; `model_old/blocks copy.py`; `model_old/blocks.py`
- `model copy/config.py`; `model_old/config.py`
- `model copy/sei.py`; `model/sei.py`; `model_old/sei.py`
- `model copy/train.py`; `model_old/train.py`
- `model copy/train_emb copy.py`; `model_old/train_emb copy.py`
- `model copy/utils_sc copy.py`; `model_old/utils_sc copy.py`

## Per-file disposition

| Source file | Migration / review result |
| --- | --- |
| `.gitignore` | Replaced by portable project-level ignore/packaging configuration. |
| `.vscode/settings.json` | Excluded: editor-specific workspace settings. |
| `Readme.md` | README.md/docs: rewrite portable commands and data contract; distinguish historical recipes from tested examples. |
| `bin/chromepi_data.py` | Excluded/replaced: legacy human float16/interpolated ChIP plus optional ICE/OE orchestration differs from masked preparation; preserve existing H5 inputs for historical results. |
| `bin/cool2npy_ice.py` | Excluded: automatic normalization/diagonal cache pipeline; new preparation accepts explicit already-prepared raw/OE cool inputs without fallback. |
| `bin/generate_hic_oe.py` | Replaced by generic prepare --hic-type oe: remove accession map and relative H5 link generation. |
| `bin/get_bed_fromfasta.py` | windows.py: retain interval generation algorithm; require explicit FASTA/output. |
| `bin/get_sei_embedding.py` | embed.py: preserve chunk/mean pooling; explicit paths, no AMP, coordinates/checksums and atomic outputs. |
| `bin/inference.py` | Replaced by portable project-level ignore/packaging configuration. |
| `bin/inference_h5_to_cool.py` | Excluded: cool/bigWig export is downstream analysis, outside this release. |
| `bin/prepare_mouse_data.py` | Excluded: earlier metadata/stage/pairs orchestration overlaps newer preparation; no pairs conversion is part of training/inference. |
| `bin/prepare_mouse_dataset.py` | preparation_io.py/prepare.py: retain bounded ChIP/Hi-C readers; replace mouse-specific manifest with explicit canonical TSV. |
| `config/mouse_hic.tsv` | Excluded: dataset-specific mouse accession map; use explicit per-cell target manifest and cool URI. |
| `model copy/blocks copy.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model copy/blocks.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model copy/blocks_alpha copy.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model copy/blocks_alpha.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model copy/config.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model copy/sei.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model copy/train.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model copy/train_emb copy.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model copy/train_emb.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model copy/utils_sc copy.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model copy/utils_sc.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model/chromepi_dataset.py` | data.py/schema.py: require precomputed embeddings; remove hardcoded FASTA/augmentation fallback; keep finite masks and accessibility sampling. |
| `model/config.py` | runtime.py/config.py: consolidate seeding, provenance and run paths; refuse run reuse. |
| `model/data.py` | sampling.py/data.py: preserve sampler algorithm; unify shape/coordinate/channel validation and configurable workers. |
| `model/depth.py` | Archived sequencing-depth conditioning for the experimental model. |
| `model/layers.py` | network.py: retain full-input operations and parameter names; remove dormant ablation/depth branches and format with AST. |
| `model/sei.py` | sei.py: retain the convolutional embedding path; omit unused classifier/spline and SEI training utilities. |
| `model/train.py` | train.py/config.py/cli.py: replace local paths and broad warning suppression with strict TOML and explicit inputs. |
| `model/trainer.py` | losses.py/train.py/checkpoint.py: keep masked window weighting; strict initialization and finite-loss checks. |
| `model/utils.py` | checkpoint.py/train.py: atomic checkpoint publication; same early-stopping task preference. |
| `model_old/blocks copy.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model_old/blocks.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model_old/blocks_alpha.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model_old/blocks_alpha_extent.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model_old/blocks_unet.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model_old/config.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model_old/sei.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model_old/train.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model_old/train_emb copy.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model_old/train_emb.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model_old/utils_sc copy.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `model_old/utils_sc.py` | Excluded: archived alternative architectures/monolithic trainers and duplicated utilities; not imported by the active full-input runtime. Static parse/definitions reviewed; no runtime guarantee for archived experiments. |
| `script/convert_humanbrain_scatac_h5_to_cool_bw.sh` | Excluded: cool/bigWig export is downstream analysis, outside this release. |
| `script/convert_mouse_CHARM_scatac_h5_to_cool_bw.sh` | Excluded: cool/bigWig export is downstream analysis, outside this release. |
| `script/convert_mouse_atac_h5_to_cool_bw.sh` | Excluded: cool/bigWig export is downstream analysis, outside this release. |
| `script/convert_mouse_dnase_h5_to_cool_bw.sh` | Excluded: cool/bigWig export is downstream analysis, outside this release. |
| `script/convert_pbmc_scatac_h5_to_cool_bw.sh` | Excluded: cool/bigWig export is downstream analysis, outside this release. |
| `script/convert_six_cell_type_atac_h5_to_cool_bw.sh` | Excluded: cool/bigWig export is downstream analysis, outside this release. |
| `script/convert_six_cell_type_rna_h5_to_cool_bw.sh` | Excluded: cool/bigWig export is downstream analysis, outside this release. |
| `script/convert_six_cell_type_scatac_h5_to_cool_bw.sh` | Excluded: cool/bigWig export is downstream analysis, outside this release. |
| `script/convert_six_cell_type_scrna_h5_to_cool_bw.sh` | Excluded: cool/bigWig export is downstream analysis, outside this release. |
| `script/generate_mouse_sei_embedding.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/inference_WCT11_atac.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/inference_humanbrain_scatac.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/inference_mouse_CHARM_scatac.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/inference_mouse_atac.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/inference_mouse_dnase.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/inference_pbmc_scatac.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/inference_six_cell_type_atac.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/inference_six_cell_type_rna.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/inference_six_cell_type_scatac.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/inference_six_cell_type_scrna.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/prepare_mouse_dataset.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/process_scatac.ipynb` | Excluded: exploratory/interactive cells, local paths and optional analysis dependencies; all code cells parsed for the inventory. |
| `script/run_loop_benchmark.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/train_bulk_dnase.sh` | Excluded: machine-specific launcher or analysis wrapper; shell syntax checked, local path assumptions inventoried; portable CLI/TOML replaces launchers. |
| `script/vis_model_predict.ipynb` | Excluded: exploratory/interactive cells, local paths and optional analysis dependencies; all code cells parsed for the inventory. |
| `tests/test_mouse_dataset.py` | tests/: port relevant masked-loss, legacy metadata, row-order and preparation coverage; omit experimental branches. |
| `tests/test_target_validation.py` | tests/: retain semantic channel/checkpoint/I/O coverage; replace fake inference with real reduced-model workflow. |
| `tmp/test_infer.ipynb` | Excluded: exploratory/interactive cells, local paths and optional analysis dependencies; all code cells parsed for the inventory. |
| `tmp/tmp.ipynb` | Excluded: exploratory/interactive cells, local paths and optional analysis dependencies; all code cells parsed for the inventory. |
