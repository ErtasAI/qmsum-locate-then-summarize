"""Commit the paired-bootstrap intervals the paper quotes, as artifacts under results/intervals/.

Every interval in Sections 6.1, 6.2, 6.4 and 7.5 was produced by running
`eval/paired_bootstrap.py` and reading its stdout. The numbers are reproducible, since the
bootstrap seed is fixed and the predictions are committed, and until now nothing in results/
recorded them. `findings.md` listed that as an open item for the artifact release.

This runs a declared set of comparisons and writes one JSON per comparison, plus an INDEX.md.
It calls `paired_bootstrap.paired_bootstrap()` and `per_query_scores()` directly, so a record
cannot drift from the tool that produced the paper's numbers.

Two things the record pins that stdout did not:

- **The sha256 of each predictions file**, so a record states exactly which text it scored. A
  regenerated prediction file with a different hash invalidates its records visibly.
- **The seed and draw count**, which are defaults in `paired_bootstrap` and were therefore
  implicit in every quoted interval.

Per-query scores are cached per (predictions file, metric) within a run, so our own row is
scored once for all twelve of its BERTScore comparisons instead of twelve times. Each cached
call still scores one file's 281 predictions in a single batched call, which is what
`per_query_bertscore` does per comparison, so the values are unchanged. Verified against the
2026-07-31 pass: distilbart 0.854595424017448, ours 0.8732747851317464, diff +0.018679,
95% CI [+0.015782, +0.021649].

Usage:
    python -m eval.interval_records --metric rouge1      # CPU
    python -m eval.interval_records --metric bertscore   # GPU, loads roberta-large
    python -m eval.interval_records                      # both
"""
import argparse, hashlib, json, pathlib

from eval.paired_bootstrap import (meeting_by_query, paired_bootstrap,
                                   paired_cluster_bootstrap, per_query_scores)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "intervals"
PREDS = ROOT / "results" / "preds"
GENERATED = "2026-08-17"
OURS = "loc-w375-l12-b2000-test"

# Every baseline in Table 1 for which we hold predictions, on both metrics. The first seven are
# the margins Section 6.1 (ROUGE-1) and Section 6.4 (BERTScore) quote; the last three are the
# community rows whose BERTScore was backfilled 2026-08-17 and which no paper table covers yet.
BASELINES = [
    ("baseline-claude-sonnet-5-test", "claude-sonnet-5, zero-shot"),
    ("baseline-claude-haiku-4-5-test", "claude-haiku-4-5, zero-shot"),
    ("baseline-claude-opus-5-test", "claude-opus-5, zero-shot"),
    ("baseline-gpt-5.6-sol-test", "gpt-5.6-sol, zero-shot"),
    ("baseline-gpt-5.6-luna-test", "gpt-5.6-luna, zero-shot"),
    ("row-distilbart-test", "distilbart, community checkpoint"),
    ("row-socratic-segenc-test", "socratic-segenc, published 406M specialist"),
    ("row-bart-large-cnn-test", "bart-large-cnn, community checkpoint"),
    ("row-pegasus-test", "pegasus, community checkpoint"),
    ("row-led-base-test", "led-base, community checkpoint"),
]

COMPARISONS = [
    {"a": slug, "b": OURS, "split": "test", "metric": m,
     "label": f"ours, promoted vs {desc}",
     "paper_section": {"rouge1": "6.1", "bertscore": "6.4"}[m]
     if slug not in ("row-bart-large-cnn-test", "row-pegasus-test", "row-led-base-test")
     else "none yet (backfilled 2026-08-17)"}
    for slug, desc in BASELINES for m in ("rouge1", "bertscore")
] + [
    # 7.5's architecture result, the span-trained SegEnc test touch that ties us.
    {"a": "segenc-c8cont-e4-test", "b": OURS, "split": "test", "metric": "rouge1",
     "label": "ours, promoted vs span-trained SegEnc, 406M", "paper_section": "7.5"},
    {"a": "segenc-c8cont-e4-test", "b": OURS, "split": "test", "metric": "bertscore",
     "label": "ours, promoted vs span-trained SegEnc, 406M", "paper_section": "7.5"},
    # 6.2's protocol-exact against promoted margin.
    {"a": "m3-fusion-lfm2.5-1.2b-test", "b": OURS, "split": "test", "metric": "rouge1",
     "label": "ours, promoted vs ours, protocol-exact", "paper_section": "6.2"},
]

# Central paper comparisons. These resample 35 meetings, retaining every query within each
# sampled meeting, because queries from one transcript are not independent observations.
COMPARISONS.extend([
    {"a": "baseline-gpt-5.6-luna-test", "b": OURS, "split": "test",
     "metric": "rouge1", "resampling_unit": "meeting",
     "label": "ours, promoted vs gpt-5.6-luna, zero-shot", "paper_section": "6.1"},
    {"a": OURS, "b": "row-socratic-segenc-test", "split": "test",
     "metric": "rouge1", "resampling_unit": "meeting",
     "label": "published Socratic SegEnc predictions vs ours, promoted",
     "paper_section": "7.5"},
    {"a": "baseline-gpt-5.6-luna-test", "b": "row-socratic-segenc-test",
     "split": "test", "metric": "rouge1", "resampling_unit": "meeting",
     "label": "published Socratic SegEnc predictions vs gpt-5.6-luna, zero-shot",
     "paper_section": "7.5"},
    {"a": OURS, "b": "segenc-c8cont-e4-test", "split": "test",
     "metric": "rouge1", "resampling_unit": "meeting",
     "label": "span-trained SegEnc vs ours, promoted", "paper_section": "7.5"},
])


def sha256(path):
    """Hash JSONL content with canonical LF endings across Git checkouts."""
    content = pathlib.Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def pin_prediction_hashes(record, preds_root=PREDS):
    """Return *record* with both prediction hashes pinned to current content."""
    for side in ("a", "b"):
        record[side]["sha256"] = sha256(preds_root / f"{record[side]['run_id']}.jsonl")
    return record


def refresh_hashes_only():
    """Repair hash metadata without recomputing expensive metric scores."""
    paths = sorted(OUT.glob("*.json"))
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        pin_prediction_hashes(record)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"refreshed portable prediction hashes in {len(paths)} records")


def scores_for(cache, run_id, split, metric):
    key = (run_id, metric)
    if key not in cache:
        cache[key] = per_query_scores(PREDS / f"{run_id}.jsonl", split, metric)
    return cache[key]


def run(comparisons):
    OUT.mkdir(parents=True, exist_ok=True)
    cache, cluster_cache, records = {}, {}, []
    for c in comparisons:
        a_scores = scores_for(cache, c["a"], c["split"], c["metric"])
        b_scores = scores_for(cache, c["b"], c["split"], c["metric"])
        unit = c.get("resampling_unit", "query")
        if unit == "meeting":
            if c["split"] not in cluster_cache:
                cluster_cache[c["split"]] = meeting_by_query(c["split"])
            res = paired_cluster_bootstrap(a_scores, b_scores, cluster_cache[c["split"]])
        else:
            res = paired_bootstrap(a_scores, b_scores)
        rec = pin_prediction_hashes({
            "record_version": 1,
            "generated": GENERATED,
            "tool": "eval/interval_records.py",
            "label": c["label"],
            "paper_section": c["paper_section"],
            "metric": c["metric"],
            "split": c["split"],
            "resampling_unit": unit,
            "draws": 10000,
            "seed": 20260723,
            "a": {"run_id": c["a"], "preds": f"results/preds/{c['a']}.jsonl"},
            "b": {"run_id": c["b"], "preds": f"results/preds/{c['b']}.jsonl"},
            **res,
            "crosses_zero": res["ci95_low"] <= 0 <= res["ci95_high"],
        })
        metric_label = c["metric"] + ("-meeting" if unit == "meeting" else "")
        path = OUT / f"{metric_label}--{c['a']}--vs--{c['b']}.json"
        path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        records.append(rec)
        print(f"{c['metric']:9s} {c['a']:34s} diff {res['observed_diff']:+.6f}  "
              f"[{res['ci95_low']:+.6f}, {res['ci95_high']:+.6f}]  "
              f"{'crosses zero' if rec['crosses_zero'] else 'excludes zero'}")
    return records


def write_index(records):
    """Rebuilt from every record on disk, so a partial run does not truncate the index."""
    on_disk = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json"))]
    on_disk.sort(key=lambda r: (r["metric"], -r["observed_diff"]))
    lines = [
        "# Committed paired-bootstrap intervals",
        "",
        "One JSON per comparison, generated by `eval/interval_records.py`. The central paper",
        "records resample 35 meetings as clusters; supporting records resample queries jointly.",
        "Every interval is 10,000 draws at seed 20260723 over the 281 test queries.",
        "Each record carries the sha256 of both predictions files it scored.",
        "",
        "Every filename and row uses B minus A. The central meeting-cluster SegEnc record uses",
        "A=ours and B=span-trained directly, matching the paper's +0.93 [-0.27, +2.22]. Its",
        "query-resampled supporting record retains the inverse A/B order and is labelled clearly.",
        "",
        "Regenerate with `python -m eval.interval_records`. ROUGE runs on CPU; BERTScore loads",
        "roberta-large and wants the GPU.",
        "",
        "| metric | resampling | A | B | diff (B minus A) | 95% interval | verdict | paper |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in on_disk:
        fmt = "{:+.2f}" if r["metric"].startswith("rouge") else "{:+.4f}"
        scale = 100 if r["metric"].startswith("rouge") else 1
        lines.append(
            f"| {r['metric']} | {r.get('resampling_unit', 'query')} | "
            f"{r['a']['run_id']} | {r['b']['run_id']} | "
            f"{fmt.format(r['observed_diff'] * scale)} | "
            f"[{fmt.format(r['ci95_low'] * scale)}, {fmt.format(r['ci95_high'] * scale)}] | "
            f"{'crosses zero' if r['crosses_zero'] else 'excludes zero'} | "
            f"{r['paper_section']} |")
    (OUT / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nINDEX.md: {len(on_disk)} records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["rouge1", "bertscore"],
                    help="run only this metric; omit for both")
    ap.add_argument("--refresh-hashes-only", action="store_true",
                    help="refresh portable prediction hashes without rescoring")
    a = ap.parse_args()
    if a.refresh_hashes_only:
        refresh_hashes_only()
        return
    cmps = [c for c in COMPARISONS if a.metric is None or c["metric"] == a.metric]
    write_index(run(cmps))


if __name__ == "__main__":
    main()
