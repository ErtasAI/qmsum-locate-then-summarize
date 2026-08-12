"""The decomposition is one division, so the tests are mostly about the guards.

`survival` is the number this whole module exists to produce and the one most
likely to be quoted out of context, so every condition that makes it meaningless
has to actually suppress it rather than shade it.
"""
import json

import pytest

from eval import retrieval_attribution as RA


def _recalls(**kw):
    return {k: {"recall": v} for k, v in kw.items()}


def _items(**kw):
    """kw maps query_id to (precision, recall)."""
    return {k: {"query_id": k, "precision": p, "recall": r} for k, (p, r) in kw.items()}


# --- the decomposition itself ------------------------------------------------

def test_survival_is_the_perfect_retrieval_gap_over_the_overall_gap():
    recalls = _recalls(q1=0.0, q2=1.0)
    ours = _items(q1=(0.0, 0.0), q2=(0.0, 0.4))
    base = _items(q1=(0.0, 0.6), q2=(0.0, 0.6))
    d = RA.decompose(recalls, ours, base, metric="recall")
    assert d["gap_all"] == pytest.approx(0.4)      # 0.6 - 0.2
    assert d["gap_perfect"] == pytest.approx(0.2)  # 0.6 - 0.4, q2 only
    assert d["survival"] == pytest.approx(0.5)
    assert d["uninterpretable_because"] == []


def test_a_locator_that_explains_everything_gives_zero_survival():
    """If our score matches the baseline wherever retrieval succeeded, the whole
    gap is retrieval and none of it belongs to the summarizer."""
    recalls = _recalls(q1=0.0, q2=1.0)
    ours = _items(q1=(0.0, 0.0), q2=(0.0, 0.6))
    base = _items(q1=(0.0, 0.6), q2=(0.0, 0.6))
    assert RA.decompose(recalls, ours, base, metric="recall")["survival"] == 0.0


def test_only_perfect_recall_counts_as_perfect_retrieval():
    """0.99 recall is a miss for this purpose: one dropped gold window is exactly
    the thing being controlled for."""
    recalls = _recalls(q1=0.99, q2=1.0)
    ours = _items(q1=(0.0, 0.1), q2=(0.0, 0.4))
    base = _items(q1=(0.0, 0.6), q2=(0.0, 0.6))
    d = RA.decompose(recalls, ours, base, metric="recall")
    assert d["n_perfect_retrieval"] == 1


def test_queries_without_a_locator_recall_are_excluded_not_zeroed():
    """General queries have no gold span, so scoring them as a retrieval miss
    would fabricate 37 failures on the test split."""
    recalls = _recalls(q1=1.0)
    ours = _items(q1=(0.0, 0.4), general=(0.0, 0.1))
    base = _items(q1=(0.0, 0.6), general=(0.0, 0.6))
    assert RA.decompose(recalls, ours, base, metric="recall")["n_shared"] == 1


# --- the guards --------------------------------------------------------------

def test_a_sloping_baseline_fails_the_control_and_suppresses_survival():
    """The baseline reads the full transcript, so its score cannot depend on OUR
    locator. If it slopes across the buckets anyway, the buckets are sorted by
    query difficulty and the decomposition measures that instead."""
    recalls = _recalls(q1=0.0, q2=1.0)
    ours = _items(q1=(0.0, 0.0), q2=(0.0, 0.4))
    base = _items(q1=(0.0, 0.2), q2=(0.0, 0.9))
    d = RA.decompose(recalls, ours, base, metric="recall")
    assert d["survival"] is None
    assert "control FAILED" in d["uninterpretable_because"][0]
    assert d["gap_all"] is not None, "the raw gaps still get reported, only the ratio is withheld"


def test_a_flat_baseline_passes_the_control():
    recalls = _recalls(q1=0.0, q2=1.0)
    ours = _items(q1=(0.0, 0.0), q2=(0.0, 0.4))
    base = _items(q1=(0.0, 0.6), q2=(0.0, 0.6))
    d = RA.decompose(recalls, ours, base, metric="recall")
    assert d["control"]["spread"] == 0.0
    assert d["survival"] is not None


def test_a_gap_too_small_to_divide_by_suppresses_survival():
    """Two systems that score the same produce a ratio of two noise terms."""
    recalls = _recalls(q1=0.0, q2=1.0)
    ours = _items(q1=(0.0, 0.50), q2=(0.0, 0.50))
    base = _items(q1=(0.0, 0.51), q2=(0.0, 0.51))
    d = RA.decompose(recalls, ours, base, metric="recall")
    assert d["survival"] is None
    assert "below" in d["uninterpretable_because"][0]


def test_an_empty_perfect_retrieval_bucket_suppresses_survival():
    recalls = _recalls(q1=0.0, q2=0.5)
    ours = _items(q1=(0.0, 0.0), q2=(0.0, 0.1))
    base = _items(q1=(0.0, 0.6), q2=(0.0, 0.6))
    d = RA.decompose(recalls, ours, base, metric="recall")
    assert d["survival"] is None
    assert d["n_perfect_retrieval"] == 0


# --- estimator and naming ----------------------------------------------------

def test_f1_scoring_counts_a_total_failure_as_zero_not_missing():
    """Reusing the aggregate f1() here once lifted our reported row by 80% by
    dropping exactly the queries the system failed on."""
    assert RA.score({"precision": 0.0, "recall": 0.0}, "f1") == 0.0
    assert RA.score({"precision": 0.5, "recall": 0.5}, "f1") == pytest.approx(0.5)


def test_zero_rates_are_counted_per_bucket():
    recalls = _recalls(q1=0.0, q2=0.0, q3=1.0)
    items = _items(q1=(0.0, 0.0), q2=(0.0, 0.0), q3=(0.5, 0.5))
    z = RA.zero_rates(recalls, items, metric="f1")
    assert z["miss (0.0)"] == {"n": 2, "zero": 2, "rate": 1.0}
    assert z["hit (1.0)"] == {"n": 1, "zero": 0, "rate": 0.0}


def test_sample_size_is_in_the_filename_except_for_the_original_n60_runs():
    """An n=281 run once overwrote an n=60 row that shared its name."""
    assert RA.facts_path("sys", "judge", 281).name == "sys__judge-judge__n281.json"
    assert RA.facts_path("sys", "judge", 60).name == "sys__judge-judge.json"


def test_markdown_says_survival_is_withheld_rather_than_printing_a_number():
    recalls = _recalls(q1=0.0, q2=1.0)
    ours = _items(q1=(0.0, 0.0), q2=(0.0, 0.4))
    base = _items(q1=(0.0, 0.2), q2=(0.0, 0.9))
    res = {"ours": "a", "baseline": "b", "judge": "j", "n": 2,
           "decompositions": {"recall": RA.decompose(recalls, ours, base, "recall")},
           "zero_rates": {"ours": RA.zero_rates(recalls, ours),
                          "baseline": RA.zero_rates(recalls, base)}}
    md = RA.render_markdown(res)
    assert "Survival not reported" in md
    assert "control FAILED" in md


# --- the real artifacts on disk ----------------------------------------------

def test_the_real_run_reproduces_and_passes_its_control():
    """Guards the actual reported number, not just the arithmetic."""
    if not RA.RECALL_FILE.exists() or not RA.facts_path(
            "m3-fusion-lfm2.5-1.2b", "gpt-5.6-sol", 281).exists():
        pytest.skip("judge artifacts not on disk")
    res = RA.run("m3-fusion-lfm2.5-1.2b", "gpt-5.6-luna", "gpt-5.6-sol", 281)
    d = res["decompositions"]["recall"]
    assert d["n_shared"] == 244, "test split has 244 specific queries"
    assert d["control"]["spread"] < 0.05, "baseline must be flat across locator buckets"
    assert d["survival"] > 0.5, "the summarizer, not the locator, owns most of the gap"
