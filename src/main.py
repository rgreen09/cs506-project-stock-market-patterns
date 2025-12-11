import argparse
from typing import List

from src.config.settings import load_config
from src.pipeline.dataset_builder import DatasetBuilder
from src.pipeline.dataset_combiner import DatasetCombiner
from src.pipeline.model_trainer import ModelTrainer


DEFAULT_SUBSET_PATH = "data/test/combined_subset.csv"


def _enabled_patterns(cfg) -> List[str]:
    return [p for p, c in cfg["patterns"].items() if c.get("enabled", True)]


def _resolve_input_path(args, cfg) -> str:
    if getattr(args, "input_path", None):
        return args.input_path
    if getattr(args, "use_subset", False):
        return DEFAULT_SUBSET_PATH
    return cfg["data"]["input_path"]


def cmd_build(args):
    cfg = load_config()
    if not args.all and not args.pattern:
        raise ValueError("Must provide --pattern or --all")
    patterns = _enabled_patterns(cfg) if args.all else [args.pattern]
    builder = DatasetBuilder()
    source_path = _resolve_input_path(args, cfg)
    for pattern in patterns:
        symbols = [args.symbol] if args.symbol else cfg["symbols"]
        for symbol in symbols:
            print(f"[build] pattern={pattern} symbol={symbol} input={source_path}")
            builder.build(pattern, symbol, input_path=source_path)


def cmd_combine(args):
    combiner = DatasetCombiner()
    combiner.combine(args.pattern)


def cmd_train(args):
    trainer = ModelTrainer()
    trainer.train(args.pattern)


def cmd_run(args):
    cfg = load_config()
    if not args.all and not args.pattern:
        raise ValueError("Must provide --pattern or --all")
    patterns = _enabled_patterns(cfg) if args.all else [args.pattern]
    builder = DatasetBuilder()
    combiner = DatasetCombiner()
    trainer = ModelTrainer()
    source_path = _resolve_input_path(args, cfg)

    for pattern in patterns:
        symbols = [args.symbol] if args.symbol else cfg["symbols"]
        for symbol in symbols:
            print(f"[run] build pattern={pattern} symbol={symbol} input={source_path}")
            builder.build(pattern, symbol, input_path=source_path)
        print(f"[run] combine pattern={pattern}")
        combined_path = combiner.combine(pattern)
        print(f"[run] train pattern={pattern}")
        trainer.train(pattern, dataset_path=combined_path)


def main():
    parser = argparse.ArgumentParser(description="Unified stock pattern pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build datasets")
    build_p.add_argument("--pattern", type=str, help="Pattern name")
    build_p.add_argument("--symbol", type=str, help="Single symbol to process")
    build_p.add_argument("--input-path", type=str, help="Override input CSV path")
    build_p.add_argument(
        "--use-subset",
        action="store_true",
        help=f"Use the test subset at {DEFAULT_SUBSET_PATH}",
    )
    build_p.add_argument("--all", action="store_true", help="Build for all enabled patterns")
    build_p.set_defaults(func=cmd_build)

    combine_p = sub.add_parser("combine", help="Combine per-symbol datasets")
    combine_p.add_argument("--pattern", required=True, type=str)
    combine_p.set_defaults(func=cmd_combine)

    train_p = sub.add_parser("train", help="Train models for a pattern")
    train_p.add_argument("--pattern", required=True, type=str)
    train_p.set_defaults(func=cmd_train)

    run_p = sub.add_parser("run", help="Build+combine+train pipeline")
    run_p.add_argument("--pattern", type=str, help="Pattern name")
    run_p.add_argument("--symbol", type=str, help="Single symbol to process")
    run_p.add_argument("--input-path", type=str, help="Override input CSV path")
    run_p.add_argument(
        "--use-subset",
        action="store_true",
        help=f"Use the test subset at {DEFAULT_SUBSET_PATH}",
    )
    run_p.add_argument("--all", action="store_true", help="Process all enabled patterns")
    run_p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

