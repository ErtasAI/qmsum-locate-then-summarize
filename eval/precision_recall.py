"""ROUGE-1 decomposed into precision and recall, per system.

WHY THIS EXISTS. The scoreboard reports F-measure, which hides the mechanism by
which the frontier baselines lose it. Decomposed, the picture is unambiguous:
they buy recall with length. Claude Opus 5 reaches 0.629 recall at 0.195
precision while writing 3.8x the reference length; our system is the only one
whose precision and recall are balanced.

This resolves an apparent contradiction in the M4 judge panel. ROUGE-1 F says we
match the reference better than gpt-5.6-luna; a reference-grounded LLM judge says
the opposite on the same pair. Both claim to measure fidelity to the reference.
They disagree because they weight precision differently: F-measure penalises the
unmatched words a long summary adds, and the judge, despite being told that
material absent from the reference is not a merit, tracks recall.

Read alongside eval/judge_control.py, which shows the judge rejects the gold
reference itself 54-6 in reference-free mode. The metric and the judge are not
two readings of one quantity; only one of them is anchored to the reference.

    python -m eval.precision_recall
"""
import argparse, json, pathlib

from data.normalize import load_queries
from eval import run_judge_panel as P
from eval.scorer import _ROUGE, sentences

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "precision-recall.json"


def load_preds(path):
    return {json.loads(l)["query_id"]: json.loads(l)["prediction"]
            for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()}


def decompose(preds, refs, metric="rouge1"):
    """Mean precision / recall / F over the queries present in both.

    Averaged per query then over queries, matching eval/scorer.py, so these
    numbers sit beside the scoreboard rather than beside a corpus-level variant.
    """
    ids = sorted(set(preds) & set(refs))
    if not ids:
        return {"n": 0, "precision": None, "recall": None, "fmeasure": None,
                "median_words": None}
    p = r = f = 0.0
    for i in ids:
        s = _ROUGE.score(sentences(refs[i]), sentences(preds[i]))[metric]
        p += s.precision; r += s.recall; f += s.fmeasure
    n = len(ids)
    lens = sorted(len((preds[i] or "").split()) for i in ids)
    return {"n": n, "precision": round(p / n, 4), "recall": round(r / n, 4),
            "fmeasure": round(f / n, 4), "median_words": lens[n // 2]}


def analyse(split=P.SPLIT, metric="rouge1"):
    queries = load_queries(split)
    refs = {q["query_id"]: q["reference"] for q in queries}
    ref_lens = sorted(len(v.split()) for v in refs.values())
    systems = {P.OURS: P.ours_preds(split)}
    systems.update({m: P.theirs_preds(m, split) for m in P.PANEL})
    rows = []
    for name, path in systems.items():
        if not pathlib.Path(path).exists():
            continue
        d = decompose(load_preds(path), refs, metric)
        d["system"] = name
        d["length_vs_reference"] = (None if d["median_words"] is None
                                    else round(d["median_words"] / ref_lens[len(ref_lens) // 2], 2))
        rows.append(d)
    rows.sort(key=lambda r: -(r["fmeasure"] or 0))
    return {"split": split, "metric": metric,
            "reference_median_words": ref_lens[len(ref_lens) // 2],
            "systems": rows}


def render_markdown(a):
    L = [f"### {a['metric'].upper()} decomposed, `{a['split']}` split", "",
         "F-measure is what the scoreboard reports. Decomposed, the frontier baselines "
         "are buying recall with length: the reference median is "
         f"{a['reference_median_words']} words, and precision falls as the multiple rises.", "",
         "| System | Precision | Recall | F | Median words | x reference |",
         "|---|---|---|---|---|---|"]
    for r in a["systems"]:
        if r["fmeasure"] is None: continue
        star = "**" if r["system"] == P.OURS else ""
        L.append(f"| {star}{r['system']}{star} | {r['precision']:.4f} | {r['recall']:.4f} | "
                 f"{r['fmeasure']:.4f} | {r['median_words']} | {r['length_vs_reference']:.2f} |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=P.SPLIT, choices=["val", "test"])
    ap.add_argument("--metric", default="rouge1", choices=["rouge1", "rouge2", "rougeL"])
    a = ap.parse_args()
    res = analyse(a.split, a.metric)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = render_markdown(res)
    (ROOT / "results" / "PRECISION-RECALL.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
