"""Read-only indexed FASTA access without a native HTSlib dependency."""

from pathlib import Path

from pyfaidx import Fasta


class IndexedFasta:
    def __init__(self, path):
        if not Path(str(path) + ".fai").is_file():
            raise FileNotFoundError(f"FASTA index is required: {path}.fai")
        self._reference = Fasta(
            str(path), as_raw=True, strict_bounds=True, rebuild=False, build_index=False
        )
        self.references = tuple(self._reference.keys())

    def get_reference_length(self, chrom):
        return len(self._reference[chrom])

    def fetch(self, chrom, start, end):
        return self._reference[chrom][start:end]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._reference.close()
