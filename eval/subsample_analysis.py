"""50-example subsample analysis of an existing prediction file (CPU only).

Context: ScaleDown's published QMSum numbers (SLM 0.300, GPT-4.1 Mini 0.342,
GPT-5.4 Nano 0.335, GPT-5.4 0.318 ROUGE-1) come from an unspecified 50-example
test slice, so an exact replication is impossible. This scores an existing full
prediction file on (a) one seeded 50-example draw and (b) a many-draw
distribution of 50-example subsets, which bounds what any such slice could have
shown. Analysis only; never a leaderboard row.
"""
import argparse, json, random, statistics
from data.normalize import load_queries
from eval.scorer import _ROUGE
from eval import protocol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--thresholds", default="0.300,0.318,0.335,0.342",
                    help="comma-separated published numbers to compare draws against")
    a = ap.parse_args()

    refs = {q["query_id"]: q["reference"] for q in load_queries(a.split)}
    preds = [json.loads(l) for l in open(a.pred, encoding="utf-8")]
    per_query = [_ROUGE.score(refs[p["query_id"]], p["prediction"])["rouge1"].fmeasure
                 for p in preds if p["query_id"] in refs]
    n = len(per_query)
    if n < a.k:
        raise SystemExit(f"only {n} scoreable predictions, need k={a.k}")

    rng = random.Random(protocol.SEED)
    seeded = statistics.mean(rng.sample(per_query, a.k))
    means = sorted(statistics.mean(rng.sample(per_query, a.k)) for _ in range(a.draws))
    lo, hi = means[int(0.025 * a.draws)], means[int(0.975 * a.draws)]

    print(f"n={n} full-split R1 {statistics.mean(per_query):.4f}")
    print(f"seeded {a.k}-example draw (seed {protocol.SEED}): {seeded:.4f}")
    print(f"{a.draws} draws of {a.k}: mean {statistics.mean(means):.4f}, "
          f"95pct interval [{lo:.4f}, {hi:.4f}], min {means[0]:.4f}, max {means[-1]:.4f}")
    for t in (float(x) for x in a.thresholds.split(",")):
        frac = sum(1 for m in means if m > t) / a.draws
        print(f"draws beating {t:.3f}: {frac:.1%}")


if __name__ == "__main__":
    main()
