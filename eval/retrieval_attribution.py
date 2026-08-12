"""How much of the fact-level gap is retrieval, and how much is the summarizer.

Session 5 answered this at n=17 and got "66% of the gap survives perfect
retrieval". n=17 is too small to spend a training budget on. This re-runs the
same decomposition over every query for which both inputs exist, which is a pure
join over data already on disk: per-query locator recall from
`eval.locator_attribution`, and per-query fact markings from `eval.judge_facts`.
No API calls, no GPU, nothing re-billed.

The decomposition:

    gap_all  = baseline_score - our_score, over all specific queries
    gap_hit  = the same difference, restricted to queries where the locator
               retrieved EVERY gold window
    survival = gap_hit / gap_all

`survival` is the fraction of the gap that a perfect locator would NOT fix, so it
is the fraction attributable to the summarizer. High survival means spend the
budget on generation; low survival means spend it on retrieval.

**The control that makes this readable, and why it is not optional.** Restricting
to the perfect-retrieval bucket only licenses a conclusion about retrieval if that
bucket is not also the easy-query bucket. If the locator succeeded exactly where
the queries were easy, `gap_hit` would shrink for reasons that have nothing to do
with retrieval and the number would be an artifact. The test is the baseline's own
score across the buckets: the baseline reads the FULL transcript, so our locator's
recall is not a fact about the baseline's input at all, only about the query. If
the baseline scores flat across buckets, the buckets are not sorted by difficulty
and the comparison holds. If it slopes, `survival` is confounded and must not be
reported. `flatness` returns the spread and `decompose` refuses to interpret a run
whose control failed.

    python -m eval.retrieval_attribution --ours m3-fusion-lfm2.5-1.2b \
        --baseline gpt-5.6-luna --judge gpt-5.6-sol --n 281
"""
import argparse, json, pathlib, statistics

from eval.judge_facts import per_query_f1
from eval.locator_attribution import bucket_by_recall

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECALL_FILE = ROOT / "results" / "locator-recall-per-query.json"
FACTS_DIR = ROOT / "results" / "judge_facts"
OUT_DIR = ROOT / "results" / "retrieval_attribution"

BUCKETS = ("miss (0.0)", "partial (0-0.5)", "most (0.5-1.0)", "hit (1.0)")

# A gap this small is noise, and dividing by it turns noise into a headline.
MIN_GAP = 0.02
# Baseline spread across buckets above this fraction of the overall gap means the
# buckets are confounded by query difficulty and `survival` is not interpretable.
MAX_CONTROL_SPREAD = 0.5


def load_recalls(path=None):
    """Per-query locator recall. Only `specific` queries have one defined."""
    d = json.loads(pathlib.Path(path or RECALL_FILE).read_text(encoding="utf-8"))
    return d["per_query"]


def facts_path(system, judge, n):
    """Sample size is part of the identity of a run, not a column in the table.

    Session 5 lost an n=60 row to an n=281 run that shared its filename. The n
    suffix is only absent for the original n=60 runs, which are named without it.
    """
    tag = f"{system}__judge-{judge}" + (f"__n{n}" if n and n != 60 else "")
    return FACTS_DIR / f"{tag}.json"


def load_items(system, judge, n, facts_dir=None):
    """Per-query fact markings, keyed by query_id."""
    p = facts_path(system, judge, n)
    if facts_dir:
        p = pathlib.Path(facts_dir) / p.name
    d = json.loads(p.read_text(encoding="utf-8"))
    return {i["query_id"]: i for i in d["items"]}


def score(item, metric):
    """One per-query number. F1 uses the per-query estimator, so a total failure
    is a 0.0 and not a dropped row: see `judge_facts.per_query_f1`."""
    if metric == "f1":
        return per_query_f1(item.get("precision"), item.get("recall"))
    return item.get(metric)


def scores_by_qid(items, metric):
    return {q: score(i, metric) for q, i in items.items()}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def flatness(recalls, items, metric):
    """Control: how much the baseline's own score varies across locator buckets.

    Returns the per-bucket means plus their spread (max minus min). The baseline
    never sees our locator's output, so any slope here is query difficulty, not
    retrieval, and it contaminates the decomposition.
    """
    b = bucket_by_recall(recalls, scores_by_qid(items, metric))
    means = [b[k]["mean"] for k in BUCKETS if b[k]["mean"] is not None]
    return {"by_bucket": b,
            "spread": round(max(means) - min(means), 4) if len(means) > 1 else None,
            "stdev": round(statistics.pstdev(means), 4) if len(means) > 1 else None}


def decompose(recalls, ours, baseline, metric="recall"):
    """Split the gap into what a perfect locator would fix and what it would not.

    Returns None for `survival` rather than a number whenever it would be
    uninterpretable: too small an overall gap to divide by, an empty
    perfect-retrieval bucket, or a failed flatness control. A silent number here
    would be the most quotable thing in the file and the easiest to get wrong.
    """
    shared = [q for q in ours if q in baseline and q in recalls]
    hit = [q for q in shared if recalls[q]["recall"] == 1.0]

    o_all, b_all = (mean([score(ours[q], metric) for q in shared]),
                    mean([score(baseline[q], metric) for q in shared]))
    o_hit, b_hit = (mean([score(ours[q], metric) for q in hit]),
                    mean([score(baseline[q], metric) for q in hit]))

    gap_all = None if None in (o_all, b_all) else b_all - o_all
    gap_hit = None if None in (o_hit, b_hit) else b_hit - o_hit

    control = flatness(recalls, baseline, metric)
    reasons = []
    if gap_all is None or gap_hit is None:
        reasons.append("a bucket was empty, no gap to decompose")
    elif gap_all < MIN_GAP:
        reasons.append(f"overall gap {gap_all:.4f} is below {MIN_GAP}, the ratio is noise")
    if control["spread"] is not None and gap_all and gap_all >= MIN_GAP \
            and control["spread"] > MAX_CONTROL_SPREAD * gap_all:
        reasons.append(f"control FAILED: baseline varies {control['spread']:.4f} across "
                       f"buckets against a gap of {gap_all:.4f}, so the buckets are "
                       f"confounded by query difficulty")

    return {"metric": metric, "n_shared": len(shared), "n_perfect_retrieval": len(hit),
            "ours_all": _r(o_all), "baseline_all": _r(b_all), "gap_all": _r(gap_all),
            "ours_perfect": _r(o_hit), "baseline_perfect": _r(b_hit), "gap_perfect": _r(gap_hit),
            "survival": None if reasons else round(gap_hit / gap_all, 4),
            "uninterpretable_because": reasons,
            "control": control,
            "ours_by_bucket": bucket_by_recall(recalls, scores_by_qid(ours, metric))}


def _r(x, nd=4):
    return None if x is None else round(x, nd)


def zero_rates(recalls, items, metric="f1"):
    """Share of queries scoring exactly zero, per bucket.

    The mean hides a bimodal failure distribution, and on this system the
    bimodality is the finding: report this beside every mean.
    """
    sc = scores_by_qid(items, metric)
    out = {}
    for name in BUCKETS:
        qs = [q for q, s in sc.items()
              if s is not None and bucket_of(recalls.get(q, {}).get("recall")) == name]
        z = sum(1 for q in qs if sc[q] == 0)
        out[name] = {"n": len(qs), "zero": z,
                     "rate": round(z / len(qs), 4) if qs else None}
    return out


def bucket_of(recall):
    if recall is None: return None
    if recall == 0: return BUCKETS[0]
    if recall < 0.5: return BUCKETS[1]
    if recall < 1.0: return BUCKETS[2]
    return BUCKETS[3]


def run(ours_system, baseline_system, judge, n, metrics=("recall", "precision", "f1")):
    recalls = load_recalls()
    ours = load_items(ours_system, judge, n)
    baseline = load_items(baseline_system, judge, n)
    return {"ours": ours_system, "baseline": baseline_system, "judge": judge, "n": n,
            "decompositions": {m: decompose(recalls, ours, baseline, m) for m in metrics},
            "zero_rates": {"ours": zero_rates(recalls, ours),
                           "baseline": zero_rates(recalls, baseline)}}


def render_markdown(res):
    L = [f"# Retrieval versus generation, fact-level gap decomposition",
         "",
         f"`{res['ours']}` against `{res['baseline']}`, judge `{res['judge']}`, "
         f"n={res['n']} requested.", ""]
    for metric, d in res["decompositions"].items():
        L += [f"## fact {metric}", "",
              f"Specific queries scored: {d['n_shared']}, of which "
              f"{d['n_perfect_retrieval']} had FULL locator recall.", "",
              "| Subset | ours | baseline | gap |", "|---|---|---|---|",
              f"| all specific | {d['ours_all']} | {d['baseline_all']} | {d['gap_all']} |",
              f"| perfect retrieval | {d['ours_perfect']} | {d['baseline_perfect']} | {d['gap_perfect']} |",
              ""]
        if d["survival"] is None:
            L += [f"**Survival not reported.** " + "; ".join(d["uninterpretable_because"]), ""]
        else:
            L += [f"**{d['survival']:.1%} of the gap survives perfect retrieval**, so that "
                  f"share is attributable to the summarizer rather than the locator.", ""]
        L += ["Control, baseline score by locator bucket (the baseline reads the full "
              "transcript, so a flat row means the buckets are not sorted by query "
              "difficulty):", ""]
        L += ["| bucket | " + " | ".join(BUCKETS) + " |", "|---|" + "---|" * len(BUCKETS)]
        L += ["| baseline | " + " | ".join(
            str(d["control"]["by_bucket"][k]["mean"]) for k in BUCKETS) + " |"]
        L += ["| ours | " + " | ".join(
            str(d["ours_by_bucket"][k]["mean"]) for k in BUCKETS) + " |",
            "", f"Baseline spread across buckets: {d['control']['spread']}.", ""]
    L += ["## Queries scoring exactly zero (fact F1), by locator bucket", "",
          "| system | " + " | ".join(BUCKETS) + " |", "|---|" + "---|" * len(BUCKETS)]
    for who in ("ours", "baseline"):
        z = res["zero_rates"][who]
        L.append(f"| {res[who if who == 'ours' else 'baseline']} | " + " | ".join(
            f"{z[k]['zero']}/{z[k]['n']}" for k in BUCKETS) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="m3-fusion-lfm2.5-1.2b")
    ap.add_argument("--baseline", default="gpt-5.6-luna")
    ap.add_argument("--judge", default="gpt-5.6-sol")
    ap.add_argument("--n", type=int, default=281)
    a = ap.parse_args()
    res = run(a.ours, a.baseline, a.judge, a.n)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{a.ours}__vs__{a.baseline}__judge-{a.judge}__n{a.n}"
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = render_markdown(res)
    (OUT_DIR / "ATTRIBUTION.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {OUT_DIR / (stem + '.json')} and ATTRIBUTION.md")


if __name__ == "__main__":
    main()
