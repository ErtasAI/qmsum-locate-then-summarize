"""How much of a judge's verdict is explained by "pick the longer summary"?

Motivation. On the QMSum test split our system writes 56 words at the median
against a 59-word human reference (0.95x), while the frontier baselines write
81-223 (1.4x to 3.8x). If a reference-free judge mostly rewards length, a panel
run in that mode measures verbosity and reports it as quality.

This does not argue the point, it counts it. For every judged item we know which
summary the judge picked and how long each one was, so P(judge picked the longer
summary) is directly observable. Read it against the 0.5 a length-blind judge
would produce:

    ~0.50  the judge is indifferent to length
    ~1.00  the judge is a length detector on this task

Ties are excluded (no side was picked). Items where the two summaries are within
`tolerance` words are excluded too: "the longer one" is not meaningful when the
difference is a rounding error, and counting them would drag the rate toward 0.5
and understate the effect.

    python -m eval.judge_length_bias
    python -m eval.judge_length_bias --mode grounded
"""
import argparse, json, pathlib

from eval import run_judge_panel as P

ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_TOLERANCE = 5


def word_counts(pred_file):
    out = {}
    for line in pathlib.Path(pred_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["query_id"]] = len((r["prediction"] or "").split())
    return out


def length_bias(run, ours_lens, theirs_lens, tolerance=DEFAULT_TOLERANCE):
    """P(the judge picked the longer summary), over decided, non-near-tie items."""
    longer_picked = comparable = ours_longer_and_won = theirs_longer_and_won = 0
    for item in run["items"]:
        w = item["winner"]
        if w not in ("a", "b"):
            continue
        qid = item["query_id"]
        if qid not in ours_lens or qid not in theirs_lens:
            continue
        la, lb = ours_lens[qid], theirs_lens[qid]
        if abs(la - lb) <= tolerance:
            continue
        comparable += 1
        picked_len, other_len = (la, lb) if w == "a" else (lb, la)
        if picked_len > other_len:
            longer_picked += 1
            if w == "a": ours_longer_and_won += 1
            else: theirs_longer_and_won += 1
    return {"contestant": run["contestant"],
            "mode": run.get("mode", "preference"),
            "judge_model": run["judge_model"],
            "self_preference_risk": run["self_preference_risk"],
            "n_comparable": comparable,
            "picked_longer": longer_picked,
            "p_picked_longer": round(longer_picked / comparable, 4) if comparable else None,
            "wins_where_the_winner_was_longer": {
                "ours": ours_longer_and_won, "theirs": theirs_longer_and_won}}


def analyse(mode="preference", tolerance=DEFAULT_TOLERANCE):
    ours_lens = word_counts(P.ours_preds())
    rows = []
    for run in P.load_runs(mode=mode):
        theirs_lens = word_counts(P.theirs_preds(run["contestant"]))
        rows.append(length_bias(run, ours_lens, theirs_lens, tolerance))
    rates = [r["p_picked_longer"] for r in rows if r["p_picked_longer"] is not None]
    return {"mode": mode, "tolerance_words": tolerance, "runs": rows,
            "mean_p_picked_longer": round(sum(rates) / len(rates), 4) if rates else None,
            "length_blind_baseline": 0.5}


def render_markdown(a):
    L = [f"### Length bias, `{a['mode']}` mode", "",
         f"P(judge picked the longer summary), ties and items within "
         f"{a['tolerance_words']} words excluded. A length-blind judge scores "
         f"{a['length_blind_baseline']}.", "",
         "| Contestant | Judge | Comparable | Picked longer | P |",
         "|---|---|---|---|---|"]
    for r in a["runs"]:
        if r["p_picked_longer"] is None: continue
        flag = " *" if r["self_preference_risk"] else ""
        L.append(f"| {r['contestant']}{flag} | {r['judge_model']} | {r['n_comparable']} | "
                 f"{r['picked_longer']} | {r['p_picked_longer']:.3f} |")
    L += ["", f"- Mean: **{a['mean_p_picked_longer']}** against a length-blind 0.5",
          "- `*` = judge shares a vendor with the contestant", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=P.MODES, default=None,
                    help="default: every mode with runs on disk")
    ap.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    a = ap.parse_args()
    docs, out = [], {}
    for mode in ([a.mode] if a.mode else P.MODES):
        res = analyse(mode, a.tolerance)
        if not res["runs"]: continue
        out[mode] = res
        docs.append(render_markdown(res))
    if not out:
        print("no judge runs found; run the panel first"); return
    P.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (P.OUT_DIR / "length-bias.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = "\n\n".join(docs)
    (P.OUT_DIR / "LENGTH-BIAS.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
