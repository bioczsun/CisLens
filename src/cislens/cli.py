"""Public CLI; help does not import the numerical stack."""

import argparse
import logging

from . import __version__


def parser():
    root = argparse.ArgumentParser(
        prog="cislens", description="CisLens ATAC/DNase training and inference"
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("train", "check"):
        command = commands.add_parser(
            name,
            help="Train a model"
            if name == "train"
            else "Validate training inputs without training",
        )
        command.add_argument("--config", required=True)
        command.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
        command.add_argument("--output-dir", default=None)
    command = commands.add_parser("infer", help="Predict epigenomic tracks and local Hi-C windows")
    for name in ("checkpoint", "signal", "bed", "embedding-dir", "output"):
        command.add_argument(f"--{name}", required=True)
    command.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    command.add_argument("--batch-size", type=int, default=4)
    command.add_argument("--workers", type=int, default=0)
    command.add_argument("--device", default="auto")
    command.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    command.add_argument("--overwrite", action="store_true")
    command.add_argument("--strict-metadata", action="store_true")
    command = commands.add_parser("windows", help="Generate default 2 Mb BED windows")
    command.add_argument("--fasta", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--species", choices=("human", "mouse"), default="human")
    command.add_argument("--stride-bins", type=int, default=50)
    command.add_argument("--seed", type=int, default=1401)
    command.add_argument("--val-chr", default="chr10")
    command.add_argument("--test-chr", default="chr15")
    command.add_argument("--overwrite", action="store_true")
    command = commands.add_parser("embed", help="Generate DNA embeddings with the SEI encoder")
    for name in ("bed", "fasta", "checkpoint", "output-dir"):
        command.add_argument(f"--{name}", required=True)
    command.add_argument("--device", default="auto")
    command.add_argument("--inner-batch", type=int, default=8)
    command.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    command.add_argument("--overwrite", action="store_true")
    command = commands.add_parser(
        "prepare", help="Build supervision from ChIP bigWigs and an optional cool"
    )
    for name in ("bed", "targets", "output-dir"):
        command.add_argument(f"--{name}", required=True)
    command.add_argument("--hic", default=None)
    command.add_argument("--hic-type", choices=("raw", "oe"), default="raw")
    command.add_argument("--balance", action="store_true")
    command.add_argument("--overwrite", action="store_true")
    command = commands.add_parser("smoke", help="Run a tiny real CPU model on synthetic inputs")
    command.add_argument("--output-dir", required=True)
    commands.add_parser("doctor", help="Print the current numerical environment")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    options = vars(args).copy()
    command = options.pop("command")
    if command in {"train", "check"}:
        from pathlib import Path
        from .config import load_config
        from .train import run

        config, geometry = load_config(args.config)
        if args.device:
            config.device = args.device
        if args.output_dir:
            config.output_dir = str(Path(args.output_dir).resolve())
        print(run(config, geometry, check_only=command == "check"))
    elif command == "doctor":
        import json
        from .runtime import environment

        print(json.dumps(environment(), indent=2))
    elif command == "windows":
        from .runtime import atomic_output
        from .windows import build_intervals

        frame = build_intervals(
            args.fasta,
            args.species,
            2048,
            1024,
            args.stride_bins,
            args.seed,
            args.val_chr,
            args.test_chr,
        )
        if not (frame.split == "train").any():
            raise ValueError("No training windows in selected references")
        with atomic_output(args.output, args.overwrite) as temporary:
            frame.to_csv(temporary, sep="\t", header=False, index=False)
        print(frame.groupby("split").size())
    elif command == "smoke":
        from .smoke import run

        print(run(args.output_dir))
    else:
        from importlib import import_module

        module = import_module(f"cislens.{command}")
        print(module.run(**options))


if __name__ == "__main__":
    main()
