"""Re-aggregate every fact-level row under the corrected estimator.

WHY. Rows written before 2026-07-27 carry `fact_f1` computed as F1 of the mean
precision and mean recall. Two things were wrong with that:

  1. `eval/scorer.py` averages PER-QUERY f-measure for ROUGE. Comparing a fact F1
     built the other way against a ROUGE built that way is not like-for-like, and
     every fact-vs-ROUGE ratio in the first draft of the M4 findings mixed the two.
  2. The first attempt at a fix reused `f1()` per query, which returns None when
     precision + recall == 0. That DROPPED every total-failure query instead of
     scoring it zero, averaging over only the queries a system managed to score on.
     It inflated our own row from 0.145 to 0.262 and flattered the weakest systems
     most, because the bias scales with failure rate (frontier rows drop 5-8 of 60,
     ours drops 35, distilbart 49).

No API calls: every per-query precision/recall is already stored in `items`, so this
is pure re-aggregation. Dry run by default, exactly like `eval/rescore_rouge.py`.

    python -m eval.rescore_facts            # show what would change
    python -m eval.rescore_facts --apply    # rewrite the rows
"""
import argparse, json, pathlib

from eval import judge_facts as F

OUT_DIR = F.OUT_DIR
CONTROL_DIR = pathlib.Path(F.ROOT) / "results" / "judge_controls"


def rows():
    for f in sorted(OUT_DIR.glob("*.json")):
        if f.name in ("panel-summary.json", "FACTS.md"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if "items" in d and "system" in d:
            yield f, d


def rescore(apply=False):
    changed = []
    for f, d in rows():
        new = F.aggregate(d["system"], d["judge_model"], d["judge_provider"],
                          d["items"], d.get("unparsed", 0))
        old_f1 = d.get("fact_f1")
        # preserve run metadata the aggregate does not carry
        for k in ("split", "n_requested"):
            if k in d:
                new[k] = d[k]
        zeros = sum(1 for i in d["items"]
                    if F.per_query_f1(i["precision"], i["recall"]) == 0.0)
        changed.append({"file": f.name, "system": d["system"],
                        "judge": d["judge_model"], "n": d["n_scored"],
                        "old_fact_f1": old_f1, "new_fact_f1": new["fact_f1"],
                        "f1_of_means": new["fact_f1_of_means"],
                        "zero_queries": zeros})
        if apply:
            f.write_text(json.dumps(new, indent=2), encoding="utf-8")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the rows; without it this is a dry run")
    a = ap.parse_args()
    changed = rescore(a.apply)
    if not changed:
        print("no fact rows found"); return
    def fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    w = max(len(c["system"]) for c in changed)
    print(f"{'system':{w}} {'judge':16} {'n':>4} {'old':>8} {'new':>8} {'delta':>8} {'zeros':>9}")
    for c in sorted(changed, key=lambda c: -(c["new_fact_f1"] or 0)):
        old, new = c["old_fact_f1"], c["new_fact_f1"]
        delta = "n/a" if old is None or new is None else f"{new - old:+.4f}"
        print(f"{c['system']:{w}} {c['judge']:16} {c['n']:4d} "
              f"{fmt(old):>8} {fmt(new):>8} {delta:>8} "
              f"{c['zero_queries']:4d}/{c['n']:<4d}")
    print()
    print("DRY RUN, nothing written. Re-run with --apply." if not a.apply
          else f"rewrote {len(changed)} rows")


if __name__ == "__main__":
    main()
