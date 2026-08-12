"""Per-query locator recall, and what it costs downstream.

The aggregate locator recall@3000w is 0.612, which says nothing about WHICH
queries fail or what the failure costs. This computes recall per query so it can
be joined against per-query scores, turning "the locator is the bottleneck" from
an oracle-gap inference into a per-query attribution.

The specific thing it is here to test: fact-level judging scores our system near
zero on some queries where ROUGE still awards partial credit, because ROUGE sees
shared topic vocabulary while the propositions are about something else. If those
are the queries where the locator missed the gold spans, then ROUGE is masking
retrieval failure and the fact metric is exposing it.

Only `specific` queries with gold spans have a locator recall defined; `general`
queries address the whole meeting and are reported separately rather than scored
as 0 or silently dropped.

    python -m eval.locator_attribution --split test
"""
import argparse, json, pathlib

from data.build_locator_data import overlaps
from data.normalize import load_meetings, load_queries
from eval import protocol
from pipeline.chunker import windows as make_windows

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "locator-recall-per-query.json"


def per_query_recall(split="test", budget=None, select_fn=None, limit=None):
    """Fraction of gold-positive windows the locator captured, per query."""
    budget = budget or protocol.SPAN_BUDGET_WORDS
    if select_fn is None:
        from pipeline.locator_crossencoder import select as select_fn
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
    rows, skipped = {}, {"general": 0, "no_gold_window": 0}
    for q in load_queries(split)[:limit]:
        if q["query_type"] != "specific" or not q["gold_spans"]:
            skipped["general"] += 1
            continue
        w = make_windows(meetings[q["meeting_id"]])
        gold = [x for x in w if overlaps(x, q["gold_spans"])]
        if not gold:
            skipped["no_gold_window"] += 1
            continue
        picked = select_fn(q["query"], w, budget)
        got = sum(1 for g in gold if any(p["start"] == g["start"] for p in picked))
        rows[q["query_id"]] = {"recall": got / len(gold), "n_gold": len(gold),
                               "n_picked": len(picked)}
    return {"split": split, "budget_words": budget, "skipped": skipped,
            "n_scored": len(rows),
            "mean_recall": round(sum(r["recall"] for r in rows.values()) / len(rows), 4)
                           if rows else None,
            "per_query": rows}


def bucket_by_recall(recalls, scores, edges=(0.0, 0.5, 1.0)):
    """Mean downstream score, bucketed by locator recall.

    `scores` maps query_id to any per-query number (fact recall, ROUGE, ...).
    Buckets are [0, 0), [0, 0.5), [0.5, 1.0), [1.0, 1.0]: a miss, a partial, a
    mostly-hit, and a clean hit, which is the shape that shows a monotone
    relationship if one exists.
    """
    buckets = {"miss (0.0)": [], "partial (0-0.5)": [], "most (0.5-1.0)": [], "hit (1.0)": []}
    for qid, s in scores.items():
        r = recalls.get(qid, {}).get("recall")
        if r is None or s is None:
            continue
        if r == 0: buckets["miss (0.0)"].append(s)
        elif r < edges[1]: buckets["partial (0-0.5)"].append(s)
        elif r < edges[2]: buckets["most (0.5-1.0)"].append(s)
        else: buckets["hit (1.0)"].append(s)
    return {k: {"n": len(v), "mean": round(sum(v) / len(v), 4) if v else None}
            for k, v in buckets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--budget", type=int, default=protocol.SPAN_BUDGET_WORDS)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    res = per_query_recall(a.split, a.budget, limit=a.limit)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "per_query"}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
