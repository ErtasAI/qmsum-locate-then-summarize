"""Contract tests for the seed-dispersion analysis.

The thing worth pinning is the verdict boundary. Section 8.4 committed to two readings
BEFORE the runs existed, and the value of that commitment is entirely in it being applied
mechanically afterwards. A verdict function that drifts, or that reads a borderline result
as whichever branch is more convenient, throws away the only thing the placeholder bought.
"""
import sys, pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.seed_spread import spread, verdict, CHECKPOINT_SPREAD_R1, DETECTION_FLOOR_R1


def _runs(*pairs):
    return [{"seed": s, "rouge1": v, "promoted": False} for s, v in pairs]


# --- spread ---------------------------------------------------------------------

def test_spread_reports_range_and_names_the_extreme_seeds():
    s = spread(_runs((101, 0.3500), (202, 0.3560), (303, 0.3520)), "rouge1")
    assert s["n"] == 3
    assert s["min_seed"] == 101 and s["max_seed"] == 202
    assert s["range"] == pytest.approx(0.0060)
    assert s["mean"] == pytest.approx(0.35267, abs=1e-5)


def test_spread_needs_two_runs():
    assert spread(_runs((101, 0.35)), "rouge1") is None


def test_spread_skips_runs_missing_the_metric():
    """BERTScore is absent on some rows. An unscored run is not a zero."""
    runs = [{"seed": 101, "rouge1": 0.35, "bertscore_f1": 0.87, "promoted": False},
            {"seed": 202, "rouge1": 0.36, "bertscore_f1": None, "promoted": False},
            {"seed": 303, "rouge1": 0.34, "bertscore_f1": 0.88, "promoted": False}]
    assert spread(runs, "bertscore_f1")["n"] == 2
    assert spread(runs, "rouge1")["n"] == 3


# --- the verdict boundary, which is the point of the file -----------------------

def test_a_spread_at_or_above_the_checkpoint_spread_is_branch_a():
    name, _ = verdict(CHECKPOINT_SPREAD_R1)
    assert name == "BRANCH A"
    assert verdict(2.4)[0] == "BRANCH A"


def test_a_spread_below_the_detection_floor_is_branch_b():
    assert verdict(DETECTION_FLOOR_R1[0] - 0.01)[0] == "BRANCH B"
    assert verdict(0.2)[0] == "BRANCH B"


def test_the_band_between_floor_and_checkpoint_spread_is_not_silently_branch_b():
    """A 1.0-point seed range is comparable to the margins the paper reports.

    Letting that fall through to "seed choice is the smaller effect" would be the
    convenient reading, so it gets its own branch that still obliges disclosure.
    """
    name, why = verdict(1.0)
    assert name.startswith("BRANCH A")
    assert "detection floor" in why


def test_verdict_boundaries_are_inclusive_at_the_thresholds():
    assert verdict(DETECTION_FLOOR_R1[0])[0].startswith("BRANCH A")
    assert verdict(CHECKPOINT_SPREAD_R1)[0] == "BRANCH A"
