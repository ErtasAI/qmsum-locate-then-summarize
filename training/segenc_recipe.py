"""Load and execute the released two-stage SegEnc span-training recipe."""
import argparse
import pathlib
import subprocess
import sys

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "training-configs" / "segenc-spans-e8.yaml"


def load_recipe(path=DEFAULT_CONFIG):
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


def stage_command(recipe, stage_name, python=sys.executable):
    """Render one training stage as an argv list suitable for subprocess.run."""
    stage = next((item for item in recipe["stages"] if item["name"] == stage_name), None)
    if stage is None:
        raise ValueError(f"unknown SegEnc recipe stage: {stage_name}")
    return [
        python, "-m", "training.train_segenc",
        "--base", stage["base"],
        "--train-file", recipe["train_file"],
        "--out", recipe["out"],
        "--epochs", str(stage["epochs"]),
        "--epoch-offset", str(stage["epoch_offset"]),
        "--lr", str(stage["lr"]),
        "--grad-accum", str(recipe["grad_accum"]),
        "--max-num-chunks", str(recipe["max_num_chunks"]),
        "--warmup-ratio", str(recipe["warmup_ratio"]),
    ]


def evaluation_command(checkpoint, epoch, max_num_chunks, python="python"):
    """Render the validation command printed after each trained checkpoint."""
    return [
        python, "-m", "eval.segenc",
        "--model-dir", str(checkpoint),
        "--split", "val", "--mode", "spans", "--bf16",
        "--span-budget", "2000", "--chunk-words", "375",
        "--max-num-chunks", str(max_num_chunks),
        "--locator-ckpt", "checkpoints/locator-crossencoder-w375-l12",
        "--out", f"results/preds/segenc-spansft-e{epoch}-val.jsonl",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", help="run only the named stage; omit to run both in order")
    parser.add_argument("--dry-run", action="store_true", help="print commands without training")
    args = parser.parse_args()
    recipe = load_recipe(args.config)
    stages = [args.stage] if args.stage else [stage["name"] for stage in recipe["stages"]]
    for stage in stages:
        command = stage_command(recipe, stage)
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
