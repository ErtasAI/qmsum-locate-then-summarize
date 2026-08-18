"""Paired bootstrap over ROUGE for two prediction files on the same split.

Built 2026-07-30 for the decode sweep, where beams=2 plus a repeat penalty moved val ROUGE-1
by +0.001 and the question "is that anything?" needed an answer rather than a shrug. Also the
machinery the paper's open item 4 needs (confidence intervals on the headline margins), so it
takes an arbitrary pair of rows rather than being decode-specific.

Paired, not independent: both systems are scored on the SAME queries. The default resamples
queries jointly. ``--resample-unit meeting`` instead resamples all queries from a meeting as
one cluster, which is the paper's central inferential analysis and respects within-meeting
dependence.

Reports:
  - each system's mean and the observed difference
  - a percentile confidence interval on the DIFFERENCE
  - the fraction of draws in which B beats A, which is the directly useful quantity
  - win / loss / tie counts over individual queries

Deliberately reports an interval rather than a p-value: the decision here is "is this worth
propagating to every other row", which is a magnitude question.
"""
import argparse
import json
import pathlib
import random
import statistics

from data.normalize import load_queries
from eval.scorer import sentences, _ROUGE

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _rows(pred_path, split):
    """(query_id, prediction, reference) for every prediction whose query is in the split."""
    refs = {q["query_id"]: q["reference"] for q in load_queries(split)}
    for line in pathlib.Path(pred_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["query_id"] in refs:
            yield row["query_id"], row["prediction"], refs[row["query_id"]]


def per_query_scores(pred_path, split, metric):
    """Per-query f-measure, keyed by query_id, using the frozen scorer's own segmentation."""
    if metric == "bertscore":
        return per_query_bertscore(pred_path, split)
    out = {}
    for qid, pred, ref in _rows(pred_path, split):
        out[qid] = _ROUGE.score(sentences(ref), sentences(pred))[metric].fmeasure
    return out


def per_query_bertscore(pred_path, split):
    """Per-query BERTScore F1, keyed by query_id.

    Added 2026-07-31 to close the paper's only unqualified metric. The frontier claim is
    stated on ROUGE-1 alone because BERTScore had no intervals, while Section 6.4 reports
    BERTScore values; a metric quoted without a margin invites the reader to supply their own.

    **This must call BERTScore exactly as `eval/scorer.py` does**, one batched call with
    `lang="en"` and nothing else set, because the per-query values have to average to the
    row-level `bertscore_f1` already published for these same files. Any extra argument here
    (a pinned model name, rescaling with baseline) would produce a defensible number that
    silently fails to match the tables, which is worse than having no number.
    """
    items = list(_rows(pred_path, split))
    if not items:
        return {}
    from bert_score import score as bs
    _, _, f1 = bs([p for _, p, _ in items], [r for _, _, r in items],
                  lang="en", verbose=False)
    return {qid: float(v) for (qid, _, _), v in zip(items, f1)}


def paired_bootstrap(a_scores, b_scores, draws=10000, seed=20260723, conf=95):
    shared = sorted(set(a_scores) & set(b_scores))
    if not shared:
        raise SystemExit("no shared query_ids between the two prediction files")
    rng = random.Random(seed)
    a_vals = [a_scores[q] for q in shared]
    b_vals = [b_scores[q] for q in shared]
    n = len(shared)
    diffs = []
    b_wins = 0
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        a_mean = sum(a_vals[i] for i in idx) / n
        b_mean = sum(b_vals[i] for i in idx) / n
        diffs.append(b_mean - a_mean)
        if b_mean > a_mean:
            b_wins += 1
    diffs.sort()
    lo_i = int((100 - conf) / 2 / 100 * draws)
    hi_i = int((1 - (100 - conf) / 2 / 100) * draws) - 1
    return {
        "n_shared": n,
        "a_mean": sum(a_vals) / n,
        "b_mean": sum(b_vals) / n,
        "observed_diff": sum(b_vals) / n - sum(a_vals) / n,
        f"ci{conf}_low": diffs[lo_i],
        f"ci{conf}_high": diffs[hi_i],
        "p_b_better": b_wins / draws,
        "per_query_b_wins": sum(1 for x, y in zip(a_vals, b_vals) if y > x),
        "per_query_a_wins": sum(1 for x, y in zip(a_vals, b_vals) if x > y),
        "per_query_ties": sum(1 for x, y in zip(a_vals, b_vals) if x == y),
        "median_abs_per_query_diff": statistics.median(
            abs(y - x) for x, y in zip(a_vals, b_vals)),
    }


def paired_cluster_bootstrap(a_scores, b_scores, cluster_by_query, draws=10000,
                             seed=20260723, conf=95):
    """Paired bootstrap that samples clusters and retains every query within each one.

    Cluster draws are equally likely at the meeting level. The statistic within each draw is
    still the mean over queries, so a sampled meeting contributes all of its query-level
    differences and retains its original within-meeting query weighting.
    """
    shared = sorted(set(a_scores) & set(b_scores))
    if not shared:
        raise SystemExit("no shared query_ids between the two prediction files")
    missing = [query_id for query_id in shared if query_id not in cluster_by_query]
    if missing:
        raise SystemExit(f"missing cluster ids for {len(missing)} shared queries")

    grouped = {}
    for query_id in shared:
        grouped.setdefault(cluster_by_query[query_id], []).append(
            b_scores[query_id] - a_scores[query_id])
    cluster_ids = sorted(grouped)
    rng = random.Random(seed)
    diffs, b_wins = [], 0
    for _ in range(draws):
        sampled = [grouped[cluster_ids[rng.randrange(len(cluster_ids))]]
                   for _ in cluster_ids]
        values = [difference for cluster in sampled for difference in cluster]
        difference = sum(values) / len(values)
        diffs.append(difference)
        if difference > 0:
            b_wins += 1
    diffs.sort()
    lo_i = int((100 - conf) / 2 / 100 * draws)
    hi_i = int((1 - (100 - conf) / 2 / 100) * draws) - 1
    observed = [b_scores[q] - a_scores[q] for q in shared]
    return {
        "n_shared": len(shared),
        "n_clusters": len(cluster_ids),
        "resampling_unit": "meeting",
        "a_mean": sum(a_scores[q] for q in shared) / len(shared),
        "b_mean": sum(b_scores[q] for q in shared) / len(shared),
        "observed_diff": sum(observed) / len(observed),
        f"ci{conf}_low": diffs[lo_i],
        f"ci{conf}_high": diffs[hi_i],
        "p_b_better": b_wins / draws,
        "per_query_b_wins": sum(1 for value in observed if value > 0),
        "per_query_a_wins": sum(1 for value in observed if value < 0),
        "per_query_ties": sum(1 for value in observed if value == 0),
        "median_abs_per_query_diff": statistics.median(abs(value) for value in observed),
    }


def meeting_by_query(split):
    """Return the benchmark's query-to-meeting cluster assignment for a split."""
    return {query["query_id"]: query["meeting_id"] for query in load_queries(split)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline prediction file")
    ap.add_argument("--b", required=True, help="candidate prediction file")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--metric", default="rouge1",
                    choices=["rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore"],
                    help="bertscore needs a GPU and loads roberta-large; the others are CPU")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--resample-unit", choices=["query", "meeting"], default="query",
                    help="paper headline intervals use meeting-cluster resampling")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    a_scores = per_query_scores(a.a, a.split, a.metric)
    b_scores = per_query_scores(a.b, a.split, a.metric)
    if a.resample_unit == "meeting":
        res = paired_cluster_bootstrap(a_scores, b_scores, meeting_by_query(a.split),
                                       draws=a.draws)
    else:
        res = paired_bootstrap(a_scores, b_scores, draws=a.draws)
    res = {"label": a.label, "metric": a.metric, "a": a.a, "b": a.b, **res}
    print(json.dumps(res, indent=2))
    print(f"\n{a.metric}: A {res['a_mean']:.4f} -> B {res['b_mean']:.4f}  "
          f"diff {res['observed_diff']:+.4f}  "
          f"95% CI [{res['ci95_low']:+.4f}, {res['ci95_high']:+.4f}]  "
          f"P(B better) {res['p_b_better']:.3f}")
    crosses_zero = res["ci95_low"] <= 0 <= res["ci95_high"]
    print("VERDICT:", "difference unresolved; interval crosses zero at 95%" if crosses_zero
          else "difference excludes zero at 95%")


if __name__ == "__main__":
    main()
