"""The control calibrates the judge, so its fixture has to be exactly right:
the reference verbatim, owned by no vendor, and never mistaken for a system row."""
import json
import pytest
from data.normalize import load_queries
from eval import judge_control as C
from eval import run_judge_panel as P
from eval.judge import vendor_of_preds


def test_reference_fixture_is_the_gold_reference_verbatim(tmp_path):
    path = C.build_reference_preds("test", tmp_path / "ref-{split}.jsonl")
    refs = {q["query_id"]: q["reference"] for q in load_queries("test")}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == len(refs)
    assert all(r["prediction"] == refs[r["query_id"]] for r in rows)


def test_reference_fixture_belongs_to_no_vendor(tmp_path):
    """If the control fixture carried a snapshot, the judge run would be flagged
    for self-preference against a human-written text."""
    path = C.build_reference_preds("test", tmp_path / "ref-{split}.jsonl")
    assert vendor_of_preds(path) is None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert all("model_snapshot" not in r for r in rows)


def test_control_fixture_lives_outside_the_predictions_dir():
    """It must never be globbed as a system row by the scoreboard or chart builder."""
    assert P.PREDS not in C.CONTROL_DIR.parents and C.CONTROL_DIR != P.PREDS
    assert "judge_controls" in str(C.REFERENCE_AS_SYSTEM)


def test_control_covers_the_same_sample_the_panel_uses(tmp_path):
    """Same split and n means the control and the panel judge identical items."""
    path = C.build_reference_preds("test", tmp_path / "ref-{split}.jsonl")
    ref_ids = {json.loads(l)["query_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    ours_ids = {json.loads(l)["query_id"] for l in P.ours_preds().read_text(encoding="utf-8").splitlines() if l.strip()}
    assert ref_ids == ours_ids


def _run(mode, a_wins, b_wins, ties=0, judge="gpt-5.6-sol"):
    return {"contestant": "gpt-5.6-luna", "mode": mode, "judge_model": judge,
            "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
            "n_decided": a_wins + b_wins + ties}


def test_reference_win_rate_is_computed_over_decided_items():
    s = C.summarize_control([_run("grounded", 55, 5)])
    assert s["runs"][0]["reference_win_rate"] == pytest.approx(55 / 60, abs=1e-4)


def test_reference_win_rate_is_none_when_nothing_decided():
    assert C.summarize_control([_run("grounded", 0, 0)])["runs"][0]["reference_win_rate"] is None


def test_markdown_states_what_a_failed_control_means():
    md = C.render_markdown(C.summarize_control([_run("grounded", 30, 30)]))
    assert "not measuring reference-matching" in md
