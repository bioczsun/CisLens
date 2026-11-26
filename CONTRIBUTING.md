# Contributing

Use the reference environment described in the [README](README.md) and install
the preparation and development dependencies with `.[prepare,dev]`. Preserve
the model and data conventions in the
[reproducibility record](docs/reproducibility.md).

For changes to the implementation, run `python -m pytest -q`,
`cislens smoke --output-dir <new-directory>` and `ruff check src tests`. Verify
packaging changes by installing and testing the wheel from outside the
checkout. Use regression tests to document changes in numerical behavior or
checkpoint compatibility.

Document the change, validation results, and any implications for existing
checkpoints, preprocessing or experiment settings. Store datasets, checkpoints,
credentials and generated outputs outside version control.
