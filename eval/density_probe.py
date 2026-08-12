"""Free screening for the density probe, before anything is sent to a judge.

The hypothesis under test is that our system under-packs: the QMSum reference
fits 7.9 facts into 59 words while we fit 4.3 claims into 56, so we are ~1.7x
less dense at the same length. If simply emitting more content raises fact
recall, density is the lever and the training design follows from it. If it does
not, the lever is elsewhere and we escalate to preference optimization.

`min_new_tokens` is the crude way to force more content, and it has a specific
failure mode that would make the whole probe uninterpretable: a model pushed past
its natural stopping point often repeats rather than adds. Repetition is longer
without being denser, so marking such a run would answer the wrong question and
cost money to do it. **Everything here is free** (no API, no GPU) and exists to
decide whether the paid marking step is worth running at all.

Degeneracy gate: `repeat_4gram` is the fraction of 4-grams in a summary that
already appeared earlier in the SAME summary. led-base, the known-degenerate row
on this benchmark, is the reference point for what bad looks like.

    python -m eval.density_probe --split val \
        --baseline m3-fusion-lfm2.5-1.2b --variants density-min150 density-min250
"""
import argparse, collections, json, pathlib, re, statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREDS = ROOT / "results" / "preds"
OUT = ROOT / "results" / "density_probe"

# Above this share of self-repeated 4-grams a summary is padding, not content.
# led-base sits far above it; healthy rows sit near zero.
DEGENERATE_REPEAT_4GRAM = 0.20


def load_preds(name, split):
    """Accepts a bare system name or an explicit path."""
    p = pathlib.Path(name)
    if not p.exists():
        p = PREDS / f"{name}-{split}.jsonl"
    return {json.loads(l)["query_id"]: json.loads(l)["prediction"]
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def words(text):
    return re.findall(r"[\w']+", text.lower())


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def repeat_4gram(text):
    """Fraction of 4-grams that already occurred earlier in the same summary."""
    w = words(text)
    if len(w) < 4:
        return 0.0
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    seen, repeats = set(), 0
    for g in grams:
        if g in seen:
            repeats += 1
        seen.add(g)
    return repeats / len(grams)


def type_token_ratio(text):
    w = words(text)
    return len(set(w)) / len(w) if w else 0.0


def profile(preds, refs=None):
    """Shape of a prediction set. `claims` is deliberately absent: counting
    propositions needs the judge, and the whole point here is to decide whether
    to pay for that."""
    wc = [len(words(t)) for t in preds.values()]
    sc = [len(sentences(t)) for t in preds.values()]
    rep = [repeat_4gram(t) for t in preds.values()]
    ttr = [type_token_ratio(t) for t in preds.values()]
    out = {"n": len(preds),
           "words_median": round(statistics.median(wc), 1),
           "words_mean": round(statistics.mean(wc), 1),
           "sentences_median": round(statistics.median(sc), 1),
           "repeat_4gram_mean": round(statistics.mean(rep), 4),
           "degenerate_rows": sum(1 for r in rep if r > DEGENERATE_REPEAT_4GRAM),
           "type_token_ratio_mean": round(statistics.mean(ttr), 4)}
    if refs:
        shared = [q for q in preds if q in refs]
        out["ref_words_median"] = round(
            statistics.median([len(words(refs[q])) for q in shared]), 1)
        out["length_vs_reference"] = round(
            statistics.mean([len(words(preds[q])) / max(1, len(words(refs[q])))
                             for q in shared]), 3)
    return out


def verdict(base, var):
    """Whether this variant is worth paying a judge to mark.

    Longer-and-repetitive answers a different question than longer-and-denser,
    so it is stopped here rather than after it has cost money.
    """
    reasons = []
    if var["degenerate_rows"] > 0.05 * var["n"]:
        reasons.append(f"{var['degenerate_rows']}/{var['n']} rows exceed the "
                       f"{DEGENERATE_REPEAT_4GRAM} self-repeat threshold")
    if var["repeat_4gram_mean"] > 2 * max(base["repeat_4gram_mean"], 0.005):
        reasons.append(f"mean self-repeat {var['repeat_4gram_mean']} is more than "
                       f"double the baseline's {base['repeat_4gram_mean']}")
    if var["words_median"] <= base["words_median"] * 1.15:
        reasons.append(f"length barely moved ({base['words_median']} -> "
                       f"{var['words_median']} words), so it does not test the hypothesis")
    return {"worth_marking": not reasons, "blocked_because": reasons}


def run(baseline, variants, split, refs=None):
    base_preds = load_preds(baseline, split)
    rows = {baseline: profile(base_preds, refs)}
    verdicts = {}
    for v in variants:
        rows[v] = profile(load_preds(v, split), refs)
        verdicts[v] = verdict(rows[baseline], rows[v])
    return {"split": split, "baseline": baseline, "profiles": rows, "verdicts": verdicts}


def render_markdown(res):
    cols = ["n", "words_median", "sentences_median", "repeat_4gram_mean",
            "degenerate_rows", "type_token_ratio_mean"]
    L = [f"# Density probe, free screening ({res['split']} split)", "",
         "Decides whether the paid fact-marking step is worth running. Counting "
         "propositions needs a judge; these proxies do not.", "",
         "| run | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, p in res["profiles"].items():
        L.append(f"| {name} | " + " | ".join(str(p.get(c, "")) for c in cols) + " |")
    L += ["", "## Verdicts", ""]
    for v, d in res["verdicts"].items():
        if d["worth_marking"]:
            L.append(f"- **{v}: worth marking.** Longer without degenerating.")
        else:
            L.append(f"- **{v}: DO NOT MARK.** " + "; ".join(d["blocked_because"]))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--baseline", default="m3-fusion-lfm2.5-1.2b")
    ap.add_argument("--variants", nargs="+", required=True)
    a = ap.parse_args()
    from data.normalize import load_queries
    refs = {q["query_id"]: q["reference"] for q in load_queries(a.split)}
    res = run(a.baseline, a.variants, a.split, refs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"screening-{a.split}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = render_markdown(res)
    (OUT / f"SCREENING-{a.split}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
