"""Seed dispersion on the promoted configuration, against the benchmark's other variances.

The question Section 8.4 asks is not "how much does the seed move the score" on its own. It
is whether that movement is large compared with the margins this paper reports. So every
figure here is printed next to the two reference quantities already established:

  within-run checkpoint spread   1.7 ROUGE-1   (Section 3.4, five checkpoints of one run)
  paired detection floor         0.9 to 1.4    (Section 3.4, full-split paired bootstrap)

A seed spread inside the detection floor means the reported margins are safe from this
particular objection. A spread at or above the checkpoint spread means several margins in
Sections 6 and 7.5 sit inside the noise of the process that produced them, and the paper
has to say so. Section 8.4 commits to both readings in advance, which is why this script
prints the comparison rather than a verdict.

The promoted run (seed 20260723, the frozen protocol seed) is one of the samples. Its
score is comparable to the others because contention changes timing and not weights; only
its wall clock is excluded from the timing summary.

Usage: python -m eval.seed_spread
"""
import glob, json, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]

PROMOTED_ROW = "loc-w375-l12-b2000-val"
PROMOTED_SEED = 20260723
SEED_ROW_GLOB = "m3-lfm25-seed*-val"

CHECKPOINT_SPREAD_R1 = 1.7      # Section 3.4
DETECTION_FLOOR_R1 = (0.9, 1.4)  # Section 3.4
METRICS = ("rouge1", "rouge2", "rougeLsum", "bertscore_f1")


def _row(name):
    p = ROOT / "results" / "rows" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def collect():
    """Every run of the promoted configuration that differs only in training seed."""
    runs = []
    promoted = _row(PROMOTED_ROW)
    if promoted:
        runs.append({"seed": PROMOTED_SEED, "row": PROMOTED_ROW, "promoted": True, **promoted})
    for p in sorted(glob.glob(str(ROOT / "results" / "rows" / f"{SEED_ROW_GLOB}.json"))):
        d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        stem = pathlib.Path(p).stem
        seed = int(stem.split("seed")[1].split("-")[0])
        runs.append({"seed": seed, "row": stem, "promoted": False, **d})
    return runs


def spread(runs, metric):
    """Range, standard deviation and extremes, in the metric's own units."""
    vals = [(r["seed"], r[metric]) for r in runs if r.get(metric) is not None]
    if len(vals) < 2:
        return None
    nums = [v for _, v in vals]
    lo = min(vals, key=lambda kv: kv[1])
    hi = max(vals, key=lambda kv: kv[1])
    return {
        "n": len(nums),
        "min": lo[1], "min_seed": lo[0],
        "max": hi[1], "max_seed": hi[0],
        "range": hi[1] - lo[1],
        "mean": statistics.mean(nums),
        "stdev": statistics.stdev(nums) if len(nums) > 1 else 0.0,
    }


def training_records():
    recs = []
    for p in sorted(glob.glob(str(ROOT / "results" / "training-runs" / "m3-fusion-lfm2.5-1.2b*.json"))):
        d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        if not d.get("dry_run"):
            recs.append(d)
    return recs


def verdict(range_r1_points):
    """The reading Section 8.4 committed to in advance. Branch A and B, not a new opinion."""
    if range_r1_points >= CHECKPOINT_SPREAD_R1:
        return ("BRANCH A", "at or above the within-run checkpoint spread: a single run's score is "
                            "not a reliable estimate of this configuration, and margins in Sections "
                            "6 and 7.5 must carry the qualification")
    if range_r1_points >= DETECTION_FLOOR_R1[0]:
        return ("BRANCH A (weaker form)", "inside the checkpoint spread but at or above the paired "
                                          "detection floor, so it is comparable to the margins the "
                                          "paper reports and must be stated alongside them")
    return ("BRANCH B", "below the paired detection floor: seed choice is the smaller effect, and "
                        "the within-run checkpoint spread remains the larger and more surprising "
                        "result. Do not read a small number here as the benchmark being stable")


def main():
    runs = collect()
    print(f"runs of the promoted configuration differing only in seed: {len(runs)}\n")
    print(f"{'seed':>10}  {'R1':>7} {'R2':>7} {'RLsum':>7} {'BERTScore':>10}   row")
    for r in sorted(runs, key=lambda r: -r["rouge1"]):
        bs = f"{r['bertscore_f1']:.4f}" if r.get("bertscore_f1") is not None else "     .   "
        tag = " (promoted)" if r["promoted"] else ""
        print(f"{r['seed']:>10}  {r['rouge1']*100:7.2f} {r['rouge2']*100:7.2f} "
              f"{r['rougeLsum']*100:7.2f} {bs:>10}   {r['row']}{tag}")

    print()
    for m in METRICS:
        s = spread(runs, m)
        if not s:
            print(f"{m:>12}: fewer than two runs scored, no spread")
            continue
        scale = 100 if m.startswith("rouge") else 1
        unit = "points" if m.startswith("rouge") else ""
        print(f"{m:>12}: range {s['range']*scale:6.3f} {unit} "
              f"(seed {s['min_seed']} {s['min']*scale:.3f} to seed {s['max_seed']} {s['max']*scale:.3f}), "
              f"mean {s['mean']*scale:.3f}, sd {s['stdev']*scale:.3f}, n={s['n']}")

    r1 = spread(runs, "rouge1")
    if r1 and r1["n"] >= 2:
        pts = r1["range"] * 100
        name, why = verdict(pts)
        print(f"\nseed range on ROUGE-1: {pts:.2f} points, over {r1['n']} runs")
        print(f"  within-run checkpoint spread (Section 3.4): {CHECKPOINT_SPREAD_R1} points")
        print(f"  paired detection floor (Section 3.4):       {DETECTION_FLOOR_R1[0]} to {DETECTION_FLOOR_R1[1]} points")
        print(f"\n  {name}: {why}.")
        if r1["n"] < 4:
            print(f"\n  NOTE: {r1['n']} runs bound dispersion loosely. Report the observed range and "
                  f"the number of runs behind it, not a standard error.")

    recs = training_records()
    if recs:
        print(f"\ntraining cost, {len(recs)} instrumented run(s):")
        for d in recs:
            gpu = d.get("gpu_memory_used_mib_at_start")
            clean = "idle" if gpu is not None and gpu <= 600 else f"{gpu} MiB resident at start"
            print(f"  seed {d['seed']:>10}: {d['wall_clock_hms']}  peak {d.get('peak_vram_gb')} GB "
                  f"allocated / {d.get('peak_vram_reserved_gb')} reserved  ({clean}, "
                  f"empty_cache={d.get('empty_cache_every')})")


if __name__ == "__main__":
    main()
