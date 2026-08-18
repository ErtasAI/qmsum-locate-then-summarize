"""Prevents the public narrative from drifting behind the reviewed paper."""
import pathlib


README = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
NORMALIZED = " ".join(README.split())


def test_title_and_segenc_boundary_match_the_reviewed_paper():
    assert "Retrieved-Span Training for Efficient Query-Focused Meeting Summarization on QMSum" in README
    assert "git clone https://github.com/ErtasAI/qmsum-retrieved-span-training" in README
    assert "cd qmsum-retrieved-span-training" in README
    assert "35.30" in README and "38.60" in README
    assert "preprocessing, decoding, implementation, or checkpoint differences" in NORMALIZED


def test_uncertainty_language_does_not_claim_equivalence_or_a_seed_distribution():
    assert "statistically indistinguishable" not in README
    assert "Training-seed dispersion" not in README
    assert "does not establish equivalence" in NORMALIZED
    assert "two-run observed range" in README
    assert "2.8x fewer parameters" in README


def test_central_meeting_cluster_intervals_and_limitations_are_disclosed():
    assert "meeting-cluster bootstrap" in README
    assert "results/intervals/" in README
    assert "output length was not experimentally controlled" in NORMALIZED.lower()
    assert "human or factuality evaluation" in NORMALIZED


def test_exact_segenc_reproduction_routes_are_documented():
    assert "--include-comparisons" in README
    assert "python -m training.segenc_recipe" in README
    assert "--max-num-chunks 32" in README
    assert "--max-num-chunks 8" in README
