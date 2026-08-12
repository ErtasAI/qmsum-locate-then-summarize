"""Length-bias arithmetic. This number decides how the judge result is framed,
so it is tested against hand-built cases rather than trusted from one run."""
import pytest
from eval import judge_length_bias as LB


def _run(items, contestant="claude-opus-5", judge="gpt-5.6-sol", self_pref=False):
    return {"contestant": contestant, "judge_model": judge,
            "self_preference_risk": self_pref, "mode": "preference",
            "items": [{"query_id": q, "swap": False, "verdict": "X",
                       "winner": w, "truncated": False} for q, w in items]}


def test_a_judge_that_always_picks_the_longer_summary_scores_one():
    ours = {"q1": 50, "q2": 50, "q3": 50}
    theirs = {"q1": 200, "q2": 200, "q3": 200}
    r = LB.length_bias(_run([("q1", "b"), ("q2", "b"), ("q3", "b")]), ours, theirs)
    assert r["p_picked_longer"] == 1.0
    assert r["wins_where_the_winner_was_longer"] == {"ours": 0, "theirs": 3}


def test_a_judge_that_always_picks_the_shorter_summary_scores_zero():
    ours = {"q1": 50, "q2": 50}
    theirs = {"q1": 200, "q2": 200}
    assert LB.length_bias(_run([("q1", "a"), ("q2", "a")]), ours, theirs)["p_picked_longer"] == 0.0


def test_an_indifferent_judge_scores_a_half():
    ours = {"q1": 50, "q2": 50, "q3": 300, "q4": 300}
    theirs = {"q1": 200, "q2": 200, "q3": 100, "q4": 100}
    r = LB.length_bias(_run([("q1", "b"), ("q2", "a"), ("q3", "a"), ("q4", "b")]), ours, theirs)
    assert r["p_picked_longer"] == 0.5


def test_ties_are_excluded_rather_than_counted_as_half():
    ours, theirs = {"q1": 50, "q2": 50}, {"q1": 200, "q2": 200}
    r = LB.length_bias(_run([("q1", "b"), ("q2", "tie")]), ours, theirs)
    assert r["n_comparable"] == 1 and r["p_picked_longer"] == 1.0


def test_unparsed_items_are_excluded():
    ours, theirs = {"q1": 50, "q2": 50}, {"q1": 200, "q2": 200}
    assert LB.length_bias(_run([("q1", "b"), ("q2", None)]), ours, theirs)["n_comparable"] == 1


def test_near_equal_lengths_are_excluded_not_scored():
    """Within tolerance there is no meaningful 'longer one'; counting those items
    would pull the rate toward 0.5 and understate a real effect."""
    ours, theirs = {"q1": 100, "q2": 50}, {"q1": 103, "q2": 200}
    r = LB.length_bias(_run([("q1", "a"), ("q2", "b")]), ours, theirs, tolerance=5)
    assert r["n_comparable"] == 1 and r["p_picked_longer"] == 1.0


def test_tolerance_boundary_is_exclusive_of_equal_difference():
    ours, theirs = {"q1": 100}, {"q1": 105}
    assert LB.length_bias(_run([("q1", "b")]), ours, theirs, tolerance=5)["n_comparable"] == 0
    assert LB.length_bias(_run([("q1", "b")]), ours, theirs, tolerance=4)["n_comparable"] == 1


def test_items_missing_from_a_predictions_file_are_skipped():
    r = LB.length_bias(_run([("q1", "b"), ("qX", "b")]), {"q1": 50}, {"q1": 200})
    assert r["n_comparable"] == 1


def test_no_comparable_items_yields_none_not_zero():
    """None means "not measured"; 0.0 would read as a judge that prefers brevity."""
    r = LB.length_bias(_run([("q1", "tie")]), {"q1": 50}, {"q1": 200})
    assert r["p_picked_longer"] is None


def test_word_counts_reads_real_predictions():
    lens = LB.word_counts(LB.P.ours_preds())
    assert len(lens) == 281 and all(v > 0 for v in lens.values())


def test_markdown_states_the_length_blind_baseline():
    md = LB.render_markdown({"mode": "preference", "tolerance_words": 5,
                             "mean_p_picked_longer": 0.9,
                             "length_blind_baseline": 0.5,
                             "runs": [LB.length_bias(_run([("q1", "b")]),
                                                     {"q1": 50}, {"q1": 200})]})
    assert "0.5" in md and "longer" in md
