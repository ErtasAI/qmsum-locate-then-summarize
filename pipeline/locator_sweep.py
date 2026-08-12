"""Sweep locator variants on RECALL alone, before paying for any summarizer run.

Why recall-first. On the 237 specific val queries the decomposition splits as: our pipeline
0.3352, our gold-span oracle 0.3687, socratic-segenc 0.3810. So retrieval is worth **+3.35 R1**,
roughly three quarters of the gap to the published specialist, and it is the largest remaining
lever. A full generate-and-score pass costs 15 to 20 minutes per variant;
`locator_crossencoder.recall_at_budget` costs a fraction of that and needs no summarizer at
all. So rank variants on recall here, then spend generation on the best one or two only.

**The measurement that motivates the variants.** Annotated relevant content is a median of
**482 words** (test) while `protocol.CHUNK_WORDS` is **900**, so a window is about twice as
coarse as the thing it is trying to find. A perfect hit still spends half its window on
irrelevant text, and any gold span crossing a boundary needs two windows. Because `select()`
packs 900-word windows into a 3,000-word budget it gets exactly **three** windows, i.e. three
chances. Finer windows would give roughly ten slots in the same budget. The locator's problem is
ranking precision at a coarse granularity rather than a shortage of budget.

**Carry this caveat.** Window-level recall is a proxy for ROUGE, and the two have already come
apart once here: the budget-4000 row had higher recall and lower R1. That comparison was
confounded by train/inference length mismatch, but it is still a warning. At a FIXED window size and budget, higher recall should mean more
gold content in the prompt. Any variant that changes window size changes the prompt distribution
too, so those must be confirmed on R1 rather than trusted on recall.

Usage:
    python -m pipeline.locator_sweep --variants base l12 --split val
    python -m pipeline.locator_sweep --list
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Reranker candidates. `base` is the incumbent, kept first so every report carries its
# reference point. l12 is the 12-layer sibling of the incumbent and is also what
# Liu and Xu (2023) used for the closest published locate-then-summarize system, so it is
# the literature-matched choice as well as the cheapest upgrade.
VARIANTS = {
    "base": {
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "note": "incumbent, 6 layers, 22.7M params, recall@3000w 0.612 val / 0.628 test",
    },
    "l12": {
        "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "note": "12 layers, ~33M. Matches the ranker family in Liu and Xu (2023)",
    },
    "electra": {
        "model": "cross-encoder/ms-marco-electra-base",
        "note": "ELECTRA-base reranker, stronger class, ~110M",
    },
    "bge": {
        "model": "BAAI/bge-reranker-base",
        "note": "BGE reranker, trained on a different mixture; check licence before shipping",
    },
    # Truncation variants, added 2026-07-30. The incumbent's 512-token context sees a mean
    # 47.7% of a 900-word window (97.3% of windows are truncated), so these two attack the
    # same defect from opposite directions and distinguish truncation from granularity.
    # Documented in the paper's locator truncation analysis.
    "w375": {
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "chunk_words": 375,
        "note": "incumbent reranker, 375-word windows so a window FITS the 512-token context",
    },
    "w375-l12": {
        "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "chunk_words": 375,
        "note": "12 layers AND untruncated windows, i.e. both cheap levers together",
    },
    "bge-m3": {
        "model": "BAAI/bge-reranker-v2-m3",
        "note": "8192-token context, so 900-word windows fit WHOLE. Isolates truncation from "
                "granularity by holding window size fixed. 568M, but the locator is 0.8% of "
                "end-to-end latency so cost is irrelevant",
    },
}

BUDGETS = (2000, 3000, 4000)


def report(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_path}")
    # UTTERANCE recall first, because it is the only one comparable across window sizes.
    # Window recall is shown after the slash for continuity with the earlier bake-off numbers.
    print("\nutterance recall / window recall, by span budget")
    print(f"{'variant':11} {'winsz':>7} " + " ".join(f"{'@' + str(b):>14}" for b in BUDGETS))
    for r in rows:
        cells = " ".join(
            f"{r['utterance_recall'].get(str(b), float('nan')):7.3f}/"
            f"{r['recall'].get(str(b), float('nan')):.3f}" for b in BUDGETS)
        print(f"{r['variant']:11} {str(r['chunk_words']):>7} {cells}")
    base = next((r for r in rows if r["variant"] == "base"), None)
    if base:
        print("\ndelta vs incumbent at the protocol budget (3000w), UTTERANCE recall:")
        for r in rows:
            if r["variant"] == "base":
                continue
            d = (r["utterance_recall"].get("3000", float("nan"))
                 - base["utterance_recall"].get("3000", float("nan")))
            print(f"  {r['variant']:11} {d:+.3f}   {r['note']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["base", "l12"],
                    help=f"any of {sorted(VARIANTS)}")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="val for deciding. test only for a final report")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-train", action="store_true",
                    help="evaluate existing checkpoints without retraining")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for name, spec in VARIANTS.items():
            print(f"{name:10} {spec['model']:45} {spec['note']}")
        return

    if a.split == "test":
        print("WARNING: test is for reporting, val is for deciding. "
              "Locator selection must happen on val.")

    from pipeline import locator_crossencoder as lc

    rows = []
    for name in a.variants:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name}; choose from {sorted(VARIANTS)}")
        spec = VARIANTS[name]
        ckpt = ROOT / "checkpoints" / (f"locator-crossencoder" if name == "base"
                                      else f"locator-crossencoder-{name}")
        chunk_words = spec.get("chunk_words")   # None means the frozen protocol size
        trained = False
        if not a.skip_train and name != "base":
            train_file = None
            if chunk_words:
                from data.build_locator_data import build, path_for
                train_file = path_for("train", chunk_words)
                if not train_file.exists():
                    print(f"\n=== building locator pairs at {chunk_words}w ===", flush=True)
                    build("train", chunk_words)
                    build("val", chunk_words)
            print(f"\n=== training {name} ({spec['model']}, "
                  f"{chunk_words or 'protocol'}w windows) ===", flush=True)
            # locator_crossencoder.train() reads BASE and CKPT at module scope, so the variant
            # is injected by rebinding them rather than by editing that module. Kept here so the
            # incumbent's own module stays byte-identical and cannot drift.
            base_orig, ckpt_orig = lc.BASE, lc.CKPT
            try:
                lc.BASE, lc.CKPT = spec["model"], ckpt
                lc.train(epochs=a.epochs, batch_size=a.batch_size, train_file=train_file)
                trained = True
            finally:
                lc.BASE, lc.CKPT = base_orig, ckpt_orig

        if not ckpt.exists():
            print(f"  SKIP {name}: no checkpoint at {ckpt}")
            continue

        print(f"\n=== evaluating {name} recall on {a.split} "
              f"({chunk_words or 'protocol'}w windows) ===", flush=True)
        ckpt_orig, model_orig = lc.CKPT, lc._MODEL
        recall, utt_recall = {}, {}
        try:
            lc.CKPT, lc._MODEL = ckpt, None
            for budget in BUDGETS:
                # chunk_words MUST match what the variant trained on, or the model is scored on
                # a text length it never saw.
                r = lc.recall_at_budget(lc.select, a.split, budget, chunk_words)
                # Window-level recall is NOT comparable across window sizes (its denominator
                # moves with chunk_words), so utterance-level recall is the headline whenever a
                # variant changes granularity. Both are reported.
                u = lc.gold_utterance_recall(lc.select, a.split, budget, chunk_words)
                recall[str(budget)] = round(r, 4)
                utt_recall[str(budget)] = round(u, 4)
                print(f"  recall@{budget}w = {r:.4f} (window)   {u:.4f} (utterance)", flush=True)
        finally:
            lc.CKPT, lc._MODEL = ckpt_orig, model_orig

        rows.append({"variant": name, "model": spec["model"], "note": spec["note"],
                     "split": a.split, "trained": trained,
                     "chunk_words": chunk_words or "protocol", "recall": recall,
                     "utterance_recall": utt_recall})

    report(rows, ROOT / "results" / f"locator-sweep-{a.split}.json")


if __name__ == "__main__":
    main()
