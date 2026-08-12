"""Attribution arithmetic, tested with a stub locator so no GPU or checkpoint
is needed. The real run is a separate, slow step."""
import pytest
from eval import locator_attribution as LA
from eval import protocol


def _select_everything(query, windows, budget_words):
    return windows


def _select_nothing(query, windows, budget_words):
    return []


def test_a_locator_that_picks_everything_has_perfect_recall():
    r = LA.per_query_recall("test", select_fn=_select_everything, limit=25)
    assert r["n_scored"] > 0
    assert r["mean_recall"] == 1.0


def test_a_locator_that_picks_nothing_has_zero_recall():
    r = LA.per_query_recall("test", select_fn=_select_nothing, limit=25)
    assert r["mean_recall"] == 0.0


def test_general_queries_are_reported_as_skipped_not_scored_as_zero():
    """A general query has no gold span, so a 0 there would be a fabricated
    failure and would drag the mean down for free."""
    r = LA.per_query_recall("test", select_fn=_select_everything, limit=25)
    assert r["skipped"]["general"] > 0
    assert r["n_scored"] + r["skipped"]["general"] + r["skipped"]["no_gold_window"] == 25


def test_budget_defaults_to_the_frozen_protocol():
    r = LA.per_query_recall("test", select_fn=_select_everything, limit=5)
    assert r["budget_words"] == protocol.SPAN_BUDGET_WORDS


# --- bucketing --------------------------------------------------------------

def _recalls(**kw):
    return {k: {"recall": v} for k, v in kw.items()}


def test_buckets_split_on_miss_partial_most_and_hit():
    b = LA.bucket_by_recall(_recalls(a=0.0, b=0.3, c=0.7, d=1.0),
                            {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4})
    assert b["miss (0.0)"] == {"n": 1, "mean": 0.1}
    assert b["partial (0-0.5)"] == {"n": 1, "mean": 0.2}
    assert b["most (0.5-1.0)"] == {"n": 1, "mean": 0.3}
    assert b["hit (1.0)"] == {"n": 1, "mean": 0.4}


def test_bucket_means_average_within_the_bucket():
    b = LA.bucket_by_recall(_recalls(a=1.0, b=1.0), {"a": 0.2, "b": 0.4})
    assert b["hit (1.0)"] == {"n": 2, "mean": pytest.approx(0.3)}


def test_queries_missing_a_recall_or_a_score_are_dropped():
    b = LA.bucket_by_recall(_recalls(a=1.0), {"a": None, "b": 0.5})
    assert all(v["n"] == 0 for v in b.values())


def test_empty_bucket_reports_none_not_zero():
    """A 0.0 in an empty bucket would read as 'the system scores zero there'."""
    b = LA.bucket_by_recall(_recalls(a=1.0), {"a": 0.9})
    assert b["miss (0.0)"] == {"n": 0, "mean": None}
