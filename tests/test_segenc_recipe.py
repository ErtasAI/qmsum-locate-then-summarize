"""Regression tests for the exact two-stage SegEnc span-training recipe."""
import pathlib

from training.segenc_recipe import evaluation_command, load_recipe, stage_command


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / "training-configs" / "segenc-spans-e8.yaml"


def test_reported_recipe_is_eight_epochs_over_fifteen_effective_chunks():
    recipe = load_recipe(CONFIG)
    assert recipe["max_num_chunks"] == 8  # stride produces 8 + 7 = 15 chunks
    assert recipe["stages"] == [
        {"name": "initial", "base": "results/external/socratic-qmsum",
         "epochs": 4, "lr": 3e-5, "epoch_offset": 0},
        {"name": "continuation", "base": "checkpoints/segenc-spans-c8/epoch4",
         "epochs": 4, "lr": 1e-5, "epoch_offset": 4},
    ]


def test_continuation_command_resumes_epoch_four_with_the_lower_lr():
    command = stage_command(load_recipe(CONFIG), "continuation", python="python")
    rendered = " ".join(command)
    assert "--base checkpoints/segenc-spans-c8/epoch4" in rendered
    assert "--lr 1e-05" in rendered
    assert "--epochs 4" in rendered
    assert "--epoch-offset 4" in rendered
    assert "--max-num-chunks 8" in rendered


def test_validation_command_preserves_the_training_chunk_regime():
    rendered = " ".join(evaluation_command(
        checkpoint="checkpoints/segenc-spans-c8/epoch8", epoch=8,
        max_num_chunks=8))
    assert "--max-num-chunks 8" in rendered
    assert "--mode spans" in rendered
    assert "segenc-spansft-e8-val.jsonl" in rendered


def test_training_script_uses_absolute_epoch_numbers_and_recipe_eval_commands():
    source = (ROOT / "training" / "train_segenc.py").read_text(encoding="utf-8")
    assert "a.epoch_offset + local_epoch" in source
    assert "evaluation_command(" in source
