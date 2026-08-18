"""Tests for the tool that produces every margin the paper reports.

Every confidence interval in the paper comes out of `paired_bootstrap`, including one that
retired a claim and one that concedes a system comparison against us. It had no tests. These
pin the properties the conclusions actually rest on, and none of them need a GPU: the
BERTScore path is exercised through a stub so the dispatch and the keying are covered without
loading roberta-large.
"""
import json

import pytest

from eval.paired_bootstrap import paired_bootstrap, paired_cluster_bootstrap, per_query_scores


def _write_preds(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_identical_systems_give_a_zero_difference_and_an_interval_containing_zero():
    scores = {f"q{i}": 0.3 + 0.01 * i for i in range(40)}
    res = paired_bootstrap(scores, dict(scores))
    assert res["observed_diff"] == pytest.approx(0.0, abs=1e-12)
    assert res["ci95_low"] <= 0 <= res["ci95_high"]
    assert res["per_query_ties"] == 40


def test_a_uniform_shift_is_detected_and_the_interval_excludes_zero():
    """Every query improves by the same amount, so the paired design should resolve it
    regardless of how spread the underlying per-query scores are. This is the property that
    makes pairing worth doing on a benchmark whose between-query variance dominates."""
    a = {f"q{i}": 0.1 + 0.02 * (i % 17) for i in range(60)}
    b = {q: v + 0.05 for q, v in a.items()}
    res = paired_bootstrap(a, b)
    assert res["observed_diff"] == pytest.approx(0.05, abs=1e-9)
    assert res["ci95_low"] > 0
    assert res["p_b_better"] == 1.0
    assert res["per_query_b_wins"] == 60


def test_the_result_is_deterministic_for_a_fixed_seed():
    """Published intervals must be reproducible from the released predictions."""
    a = {f"q{i}": (i * 37 % 11) / 10 for i in range(50)}
    b = {f"q{i}": (i * 53 % 13) / 12 for i in range(50)}
    assert paired_bootstrap(a, b) == paired_bootstrap(a, b)


def test_only_shared_query_ids_are_compared():
    a = {"q1": 0.5, "q2": 0.5, "q3": 0.5}
    b = {"q2": 0.7, "q3": 0.7, "q4": 0.7}
    res = paired_bootstrap(a, b)
    assert res["n_shared"] == 2


def test_no_overlap_is_an_error_rather_than_an_empty_mean():
    """Comparing two files from different splits would otherwise divide by zero or, worse,
    silently report a mean over nothing."""
    with pytest.raises(SystemExit):
        paired_bootstrap({"q1": 0.5}, {"q9": 0.5})


def test_cluster_bootstrap_resamples_whole_meetings_and_is_deterministic():
    """All queries from a sampled meeting travel together in every bootstrap draw."""
    a = {**{f"m1_q{i}": 0.0 for i in range(10)}, "m2_q0": 1.0}
    b = {**{f"m1_q{i}": 1.0 for i in range(10)}, "m2_q0": 0.0}
    clusters = {q: q.split("_")[0] for q in a}
    got = paired_cluster_bootstrap(a, b, clusters, draws=1000, seed=7)
    assert got == paired_cluster_bootstrap(a, b, clusters, draws=1000, seed=7)
    assert got["n_shared"] == 11
    assert got["n_clusters"] == 2
    assert got["resampling_unit"] == "meeting"
    assert got["observed_diff"] == pytest.approx(9 / 11)
    assert got["ci95_low"] == pytest.approx(-1.0)
    assert got["ci95_high"] == pytest.approx(1.0)


def test_cluster_bootstrap_requires_a_cluster_for_every_shared_query():
    with pytest.raises(SystemExit, match="missing cluster ids"):
        paired_cluster_bootstrap({"q1": 0.1, "q2": 0.2},
                                 {"q1": 0.2, "q2": 0.3}, {"q1": "m1"})


def test_rouge_metric_reads_predictions_and_keys_by_query_id(tmp_path, monkeypatch):
    import eval.paired_bootstrap as pb
    monkeypatch.setattr(pb, "load_queries", lambda split: [
        {"query_id": "v1", "reference": "the team agreed to ship the remote control"},
        {"query_id": "v2", "reference": "the budget was cut by half"},
    ])
    preds = _write_preds(tmp_path, "p.jsonl", [
        {"query_id": "v1", "prediction": "the team agreed to ship the remote control"},
        {"query_id": "v2", "prediction": "completely unrelated text about nothing"},
        {"query_id": "not_in_split", "prediction": "ignored"},
    ])
    got = per_query_scores(preds, "val", "rouge1")
    assert set(got) == {"v1", "v2"}, "predictions outside the split must be dropped"
    assert got["v1"] == pytest.approx(1.0)
    assert got["v2"] < 0.3


def test_bertscore_metric_dispatches_and_preserves_query_order(tmp_path, monkeypatch):
    """The stub returns a distinct value per pair, so a mis-zip between predictions and
    scores would show up as the wrong query carrying the wrong number. That failure would be
    invisible in the aggregate mean, which is exactly why it is worth a test."""
    import eval.paired_bootstrap as pb
    monkeypatch.setattr(pb, "load_queries", lambda split: [
        {"query_id": "v1", "reference": "ref one"},
        {"query_id": "v2", "reference": "ref two"},
        {"query_id": "v3", "reference": "ref three"},
    ])
    preds = _write_preds(tmp_path, "p.jsonl", [
        {"query_id": "v1", "prediction": "pred one"},
        {"query_id": "v2", "prediction": "pred two"},
        {"query_id": "v3", "prediction": "pred three"},
    ])

    calls = {}

    def fake_bs(preds_list, refs_list, **kw):
        calls["preds"], calls["refs"], calls["kw"] = preds_list, refs_list, kw
        return None, None, [0.81, 0.82, 0.83]

    import sys, types
    mod = types.ModuleType("bert_score")
    mod.score = fake_bs
    monkeypatch.setitem(sys.modules, "bert_score", mod)

    got = per_query_scores(preds, "val", "bertscore")
    assert got == {"v1": pytest.approx(0.81), "v2": pytest.approx(0.82),
                   "v3": pytest.approx(0.83)}
    assert calls["preds"] == ["pred one", "pred two", "pred three"]
    assert calls["refs"] == ["ref one", "ref two", "ref three"]


def test_bertscore_is_called_exactly_as_the_frozen_scorer_calls_it(tmp_path, monkeypatch):
    """`eval/scorer.py` calls bert_score with lang="en" and nothing else. The per-query values
    must average to the row-level bertscore_f1 already published for the same files, so any
    extra argument here (a pinned model, rescale_with_baseline) would produce numbers that
    quietly disagree with the tables."""
    import eval.paired_bootstrap as pb
    monkeypatch.setattr(pb, "load_queries", lambda split: [
        {"query_id": "v1", "reference": "ref one"}])
    preds = _write_preds(tmp_path, "p.jsonl", [{"query_id": "v1", "prediction": "pred one"}])

    seen = {}

    def fake_bs(p, r, **kw):
        seen.update(kw)
        return None, None, [0.9]

    import sys, types
    mod = types.ModuleType("bert_score")
    mod.score = fake_bs
    monkeypatch.setitem(sys.modules, "bert_score", mod)

    per_query_scores(preds, "val", "bertscore")
    assert seen == {"lang": "en", "verbose": False}


def test_empty_prediction_file_returns_nothing_rather_than_calling_the_model(tmp_path,
                                                                            monkeypatch):
    import eval.paired_bootstrap as pb
    monkeypatch.setattr(pb, "load_queries", lambda split: [
        {"query_id": "v1", "reference": "ref"}])
    preds = _write_preds(tmp_path, "empty.jsonl", [])
    assert per_query_scores(preds, "val", "bertscore") == {}
