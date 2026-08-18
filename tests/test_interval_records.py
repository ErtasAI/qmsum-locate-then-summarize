"""Guards on the committed paired-bootstrap records in results/intervals/.

All CPU, no model loads: every assertion reads committed JSON. The point of the records is that
the paper's intervals stop depending on someone re-running a script, so the failure mode worth
catching is a record that has quietly stopped describing the files it names.
"""
import json
import pathlib

import pytest

from eval.interval_records import COMPARISONS, OUT, PREDS, pin_prediction_hashes, sha256

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROWS = ROOT / "results" / "rows"
RECORDS = sorted(OUT.glob("*.json"))

# The 2026-07-31 pass that closed the BERTScore half of the frontier claim, and the figures
# Section 6.4 quotes from it. Pinned to full precision so a silent change in the scorer, the
# bootstrap or the predictions cannot leave the paper's numbers standing.
DISTILBART_BERTSCORE = {
    "a_mean": 0.854595424017448,
    "b_mean": 0.8732747851317464,
    "observed_diff": 0.018679361114298354,
    "ci95_low": 0.01578158734107782,
    "ci95_high": 0.02164855991818304,
}

PAPER_MEETING_CLUSTER_INTERVALS = {
    ("baseline-gpt-5.6-luna-test", "loc-w375-l12-b2000-test"):
        (2.9759346868364847, 1.3922174845699773, 4.6322960111539),
    ("loc-w375-l12-b2000-test", "row-socratic-segenc-test"):
        (3.1919307764906186, 2.014250006191889, 4.472697261849334),
    ("baseline-gpt-5.6-luna-test", "row-socratic-segenc-test"):
        (6.167865463327103, 4.835800674129924, 7.445816745161782),
    ("loc-w375-l12-b2000-test", "segenc-c8cont-e4-test"):
        (0.9272858228114956, -0.27242478402524234, 2.223740445608474),
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_records_exist():
    assert RECORDS, "results/intervals/ holds no records; run python -m eval.interval_records"


def test_prediction_hash_is_portable_across_line_endings(tmp_path):
    """The same JSONL must retain its identity on Windows and POSIX checkouts."""
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    lf.write_bytes(b'{"query_id":"q1"}\n{"query_id":"q2"}\n')
    crlf.write_bytes(b'{"query_id":"q1"}\r\n{"query_id":"q2"}\r\n')
    assert sha256(lf) == sha256(crlf)


def test_pin_prediction_hashes_updates_both_sides(tmp_path):
    (tmp_path / "run-a.jsonl").write_text('{"query_id":"q1"}\n', encoding="utf-8")
    (tmp_path / "run-b.jsonl").write_text('{"query_id":"q2"}\n', encoding="utf-8")
    record = {"a": {"run_id": "run-a"}, "b": {"run_id": "run-b"}}
    pinned = pin_prediction_hashes(record, tmp_path)
    assert pinned["a"]["sha256"] == sha256(tmp_path / "run-a.jsonl")
    assert pinned["b"]["sha256"] == sha256(tmp_path / "run-b.jsonl")


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_record_pins_the_predictions_it_scored(path):
    """A regenerated predictions file must invalidate its records loudly."""
    rec = load(path)
    for side in ("a", "b"):
        preds = PREDS / f"{rec[side]['run_id']}.jsonl"
        assert preds.exists(), f"{path.name}: missing {preds.name}"
        digest = sha256(preds)
        assert digest == rec[side]["sha256"], (
            f"{path.name}: {preds.name} has changed since the record was written. "
            f"Re-run python -m eval.interval_records --metric {rec['metric']}")


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_record_carries_the_frozen_bootstrap_settings(path):
    rec = load(path)
    assert rec["draws"] == 10000
    assert rec["seed"] == 20260723
    assert rec["n_shared"] == 281
    assert rec["split"] == "test"
    assert rec["crosses_zero"] == (rec["ci95_low"] <= 0 <= rec["ci95_high"])
    assert rec["ci95_low"] <= rec["observed_diff"] <= rec["ci95_high"]


def test_every_declared_comparison_has_a_record():
    """Adding a comparison to the script without running it should fail here."""
    declared = {(c["metric"], c["a"], c["b"], c.get("resampling_unit", "query"))
                for c in COMPARISONS}
    on_disk = {(r["metric"], r["a"]["run_id"], r["b"]["run_id"],
                r.get("resampling_unit", "query"))
               for r in (load(p) for p in RECORDS)}
    assert declared <= on_disk, f"no record for: {sorted(declared - on_disk)}"


def test_index_lists_exactly_the_records_on_disk():
    index = (OUT / "INDEX.md").read_text(encoding="utf-8")
    body = [l for l in index.splitlines() if l.startswith("| ") and not l.startswith("| metric")]
    assert len(body) == len(RECORDS)


def test_distilbart_bertscore_record_matches_the_published_figures():
    rec = load(OUT / "bertscore--row-distilbart-test--vs--loc-w375-l12-b2000-test.json")
    for k, want in DISTILBART_BERTSCORE.items():
        assert rec[k] == pytest.approx(want, abs=1e-12), k
    assert not rec["crosses_zero"]


@pytest.mark.parametrize("runs,expected", PAPER_MEETING_CLUSTER_INTERVALS.items())
def test_central_paper_intervals_use_meeting_cluster_resampling(runs, expected):
    a, b = runs
    rec = load(OUT / f"rouge1-meeting--{a}--vs--{b}.json")
    observed, low, high = expected
    assert rec["resampling_unit"] == "meeting"
    assert rec["n_clusters"] == 35
    assert rec["observed_diff"] * 100 == pytest.approx(observed, abs=1e-12)
    assert rec["ci95_low"] * 100 == pytest.approx(low, abs=1e-12)
    assert rec["ci95_high"] * 100 == pytest.approx(high, abs=1e-12)


@pytest.mark.parametrize("run_id", ["row-distilbart-test", "row-bart-large-cnn-test",
                                    "row-pegasus-test", "row-led-base-test"])
def test_community_rows_carry_a_bertscore(run_id):
    """Backfilled 2026-08-17; before that the four M0 community rows were null."""
    row = load(ROWS / f"{run_id}.json")
    assert row["bertscore_f1"] is not None, f"{run_id}: bertscore_f1 is null again"
    assert 0.0 < row["bertscore_f1"] < 1.0


def test_community_row_bertscore_matches_its_interval_record():
    """The row-level figure is the mean of the per-query values the record bootstrapped.

    Tolerance is 1e-6, and the reason is arithmetic rather than laxity. `eval/scorer.py` takes
    the mean inside torch, in float32, while `paired_bootstrap` sums the same 281 values in
    Python float64. The two agree to about 6e-8 on these rows, which is float32 accumulation
    over 281 terms. A real disagreement (a different prediction set, a different scorer call)
    would be orders of magnitude larger.
    """
    for run_id in ("row-distilbart-test", "row-bart-large-cnn-test",
                   "row-pegasus-test", "row-led-base-test"):
        row = load(ROWS / f"{run_id}.json")
        rec = load(OUT / f"bertscore--{run_id}--vs--loc-w375-l12-b2000-test.json")
        assert row["bertscore_f1"] == pytest.approx(rec["a_mean"], abs=1e-6), run_id
