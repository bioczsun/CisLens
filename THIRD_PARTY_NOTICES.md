# Third-party notices

`src/cislens/sei.py` is an embedding-only adaptation of the SEI convolutional
network retained in the source ChromEPI repository. The corresponding upstream
project is [FunctionLab/sei-framework](https://github.com/FunctionLab/sei-framework),
whose [license](https://github.com/FunctionLab/sei-framework/blob/main/LICENSE.txt)
permits redistribution subject to its terms for academic and research use only.
The upstream copyright, conditions and disclaimer are retained in LICENSES/SEI.txt.
Changes remove the unused classifier/spline head and expose the same encoder
mean-pooled representation. Upstream weights are not redistributed here.

The CisLens network and runtime derive from the supplied ChromEPI working tree;
their file hashes and disposition are recorded in docs/source_manifest.json
and docs/source_audit.md. No repository-wide license was present in that source
snapshot. The authors must determine the original-contribution license and any
additional provenance obligations before an official public release.

Dependencies retain their own licenses; installing them does not transfer their
licenses to the authors' original code. This notice does not invent a manuscript
citation or grant additional rights.
