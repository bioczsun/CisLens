# Software and data availability

This repository contains the CisLens implementation, example configurations,
and a synthetic workflow for verifying installation, training and inference.
See the [README](../README.md) and [validation record](validation.md).

Reproduction of a study analysis requires the experiment-specific inputs and
settings. Record the following with the analysis.

- The repository revision and software environment.
- The source dataset accessions, reference assembly, original BED and split
  assignments, and the parameters used to generate the input bigWigs,
  supervision and SEI embeddings.
- The CisLens and SEI checkpoints, their SHA-256 checksums and the corresponding
  model configuration, input assay, target order and preprocessing conventions.
- The training or inference configuration, software and hardware versions,
  input checksums, and the analysis scripts used to generate reported results.

The study's Data Availability statement identifies the source datasets and
their access conditions. See the [reproducibility record](reproducibility.md)
for preparation and compatibility details. Cite the CisLens manuscript and
specify the repository revision used for the analysis.

The terms applicable to the SEI-derived encoder are documented in the
[third-party notices](../THIRD_PARTY_NOTICES.md).
