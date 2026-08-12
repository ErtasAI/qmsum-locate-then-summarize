"""The precision/recall decomposition explains the metric-vs-judge disagreement,
so it has to agree with the frozen scorer rather than approximate it."""
import pytest
from eval import precision_recall as PR
from eval import run_judge_panel as P
from eval.scorer import score


def test_fmeasure_matches_the_frozen_scorer():
    """If this drifts from eval/scorer.py, the decomposition is describing a
    different quantity from the one on the scoreboard."""
    from data.normalize import load_queries
    refs = {q["query_id"]: q["reference"] for q in load_queries("test")}
    preds = PR.load_preds(P.ours_preds())
    ids = sorted(set(preds) & set(refs))
    frozen = score([preds[i] for i in ids], [refs[i] for i in ids])
    assert PR.decompose(preds, refs)["fmeasure"] == pytest.approx(frozen["rouge1"], abs=1e-4)


def test_perfect_prediction_scores_one_on_both_sides():
    refs = {"q1": "the group agreed to defer the vote"}
    d = PR.decompose(dict(refs), refs)
    assert d["precision"] == 1.0 and d["recall"] == 1.0 and d["fmeasure"] == 1.0


def test_padding_a_correct_answer_holds_recall_and_drops_precision():
    """This is the frontier baselines' mechanism, in miniature."""
    refs = {"q1": "the group agreed to defer the vote"}
    terse = PR.decompose({"q1": "the group agreed to defer the vote"}, refs)
    padded = PR.decompose({"q1": "the group agreed to defer the vote " +
                                 "and discussed many other unrelated matters at length today"}, refs)
    assert padded["recall"] == terse["recall"]
    assert padded["precision"] < terse["precision"]
    assert padded["fmeasure"] < terse["fmeasure"]


def test_recall_can_win_while_fmeasure_loses():
    """The exact shape of the judge-vs-ROUGE disagreement: a longer summary that
    covers more of the reference still loses F-measure."""
    refs = {"q1": "alpha beta gamma delta"}
    short = PR.decompose({"q1": "alpha beta"}, refs)
    long_ = PR.decompose({"q1": "alpha beta gamma delta " + " ".join(f"x{i}" for i in range(30))}, refs)
    assert long_["recall"] > short["recall"]
    assert long_["fmeasure"] < short["fmeasure"]


def test_empty_overlap_reports_none_not_zero():
    assert PR.decompose({"qX": "text"}, {"q1": "ref"})["precision"] is None


def test_analyse_covers_our_system_and_every_panel_baseline():
    a = PR.analyse()
    assert {r["system"] for r in a["systems"]} == {P.OURS, *P.PANEL}
    assert all(r["n"] == 281 for r in a["systems"])


def test_our_system_is_the_only_one_not_buying_recall_with_length():
    """Guards the claim the analysis makes. If a baseline ever lands
    inside this band, the framing needs rewriting, not the test relaxing."""
    a = PR.analyse()
    by = {r["system"]: r for r in a["systems"]}
    ours = by[P.OURS]
    assert ours["length_vs_reference"] < 1.1
    assert ours["precision"] > ours["recall"] * 0.9   # balanced, not recall-skewed
    for m in P.PANEL:
        assert by[m]["length_vs_reference"] > ours["length_vs_reference"]
        assert by[m]["precision"] < ours["precision"]


def test_markdown_reports_the_reference_length_anchor():
    md = PR.render_markdown(PR.analyse())
    assert "reference median" in md and "Precision" in md
