"""M4 blind judge panel: our winner against every frontier baseline, judged twice.

DESIGN. Each pair is judged by BOTH vendors' flagships. Because our system is
vendor-neutral to both judges (an LFM2.5 fine-tune trained on QMSum gold, so no
judge is scoring its own vendor's output on our side), every pair resolves to:

  - one CLEAN verdict, from the judge whose vendor is not in the pair
  - one SAME-VENDOR verdict, from the judge whose vendor made the contestant

The clean verdict is the number that gets reported. The same-vendor verdict is
not waste: run over the identical 60 items in the identical orientation, the
difference between the two is a direct measurement of self-preference, which is
otherwise only ever disclaimed. Inter-judge agreement and Cohen's kappa fall out
of the same data.

The panel replaces the spec's Gemini spot-check, which is unavailable: the
Gemini free tier is 20 requests/day/model and the GCP credits are scope-
restricted. Two judges that are both contestants beats one judge
that is neither only because we run both over everything and report the split.

Nothing is ever re-billed. Every response caches per (pair, judge, query,
orientation); a resume calls only what is missing.

    python -m eval.run_judge_panel run          # full panel, resumable
    python -m eval.run_judge_panel run --pairs claude-opus-5 --judges gpt
    python -m eval.run_judge_panel summarize    # recompute from cached runs only
"""
import argparse, itertools, json, pathlib

from eval import judge as J
from eval import protocol

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREDS = ROOT / "results" / "preds"
OUT_DIR = ROOT / "results" / "judge"
CACHE_ROOT = ROOT / "results" / "judge_cache"

OURS = "m3-fusion-lfm2.5-1.2b"
SPLIT = "test"
PANEL = ["gpt-5.6-luna", "gpt-5.6-sol", "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]
JUDGE_PROVIDERS = ["gpt", "claude"]
MODES = ["preference", "grounded"]


def ours_preds(split=SPLIT):
    return PREDS / f"{OURS}-{split}.jsonl"

def theirs_preds(model, split=SPLIT):
    return PREDS / f"baseline-{model}-{split}.jsonl"

def run_tag(model, judge_model, mode="preference"):
    """`preference` keeps the unsuffixed name it was first written under, so
    adding the grounded arm cannot orphan an already-paid-for run."""
    base = f"{model}__judge-{judge_model}"
    return base if mode == "preference" else f"{base}__{mode}"


def make_client(provider):
    if provider == "gpt":
        from openai import OpenAI; return OpenAI()
    if provider == "claude":
        import anthropic; return anthropic.Anthropic()
    from google import genai; return genai.Client()


def run_panel(pairs=None, providers=None, modes=None, split=SPLIT, n=None,
              clients=None, quiet=False):
    """Judge every (pair, judge, mode) combination, resuming from cache."""
    pairs = pairs or PANEL
    providers = providers or JUDGE_PROVIDERS
    modes = modes or MODES
    n = n or protocol.JUDGE_SAMPLE_N
    clients = clients or {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for mode, model, provider in itertools.product(modes, pairs, providers):
        judge_model = J.DEFAULT_JUDGE_MODEL[provider]
        tag = run_tag(model, judge_model, mode)
        if provider not in clients:
            clients[provider] = make_client(provider)
        if not quiet:
            print(f"[judge] {mode}: {OURS} vs {model}  <-  {judge_model} (n={n})", flush=True)
        tally = J.run_judgement(
            ours_preds(split), theirs_preds(model, split), split, n,
            clients[provider], provider=provider, model=judge_model,
            cache_dir=CACHE_ROOT / tag, mode=mode)
        tally["contestant"] = model
        tally["ours"] = OURS
        tally["split"] = split
        (OUT_DIR / f"{tag}.json").write_text(json.dumps(tally, indent=2), encoding="utf-8")
        if not quiet:
            print(f"         ours {tally['a_wins']} / theirs {tally['b_wins']} / "
                  f"tie {tally['ties']} / unparsed {tally['unparsed']}  "
                  f"${tally['cost_usd']:.2f}"
                  f"{'  [SELF-PREFERENCE]' if tally['self_preference_risk'] else ''}",
                  flush=True)
        runs.append(tally)
    return runs


def load_runs(pairs=None, providers=None, mode="preference"):
    pairs = pairs or PANEL
    providers = providers or JUDGE_PROVIDERS
    runs = []
    for model, provider in itertools.product(pairs, providers):
        f = OUT_DIR / f"{run_tag(model, J.DEFAULT_JUDGE_MODEL[provider], mode)}.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            r.setdefault("mode", mode)
            runs.append(r)
    return runs


def win_rate(tally, side):
    """Share of DECIDED items won by `side`. Unparsed items are excluded from the
    denominator rather than counted as draws, so a malformed response cannot
    quietly move a win rate toward 50%."""
    d = tally["n_decided"]
    return None if not d else tally[f"{side}_wins"] / d


def cohens_kappa(pairs):
    """Agreement between two judges on 3 categories, corrected for chance."""
    if not pairs: return None
    cats = ["a", "b", "tie"]
    n = len(pairs)
    observed = sum(1 for x, y in pairs if x == y) / n
    expected = sum((sum(1 for x, _ in pairs if x == c) / n) *
                   (sum(1 for _, y in pairs if y == c) / n) for c in cats)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 4)


def agreement(run_a, run_b):
    """Item-level agreement between two judges over the same sample.

    Both judges see the identical sample in the identical orientation (the
    sample is seed-deterministic and the swap is index-parity), so items are
    directly comparable by query_id.
    """
    a = {i["query_id"]: i["winner"] for i in run_a["items"]}
    b = {i["query_id"]: i["winner"] for i in run_b["items"]}
    both = [(a[q], b[q]) for q in sorted(set(a) & set(b))
            if a[q] is not None and b[q] is not None]
    if not both: return {"n_comparable": 0, "raw": None, "kappa": None}
    return {"n_comparable": len(both),
            "raw": round(sum(1 for x, y in both if x == y) / len(both), 4),
            "kappa": cohens_kappa(both)}


def summarize(runs):
    """Per pair: the clean verdict, the same-vendor verdict, and the gap."""
    by_pair = {}
    for r in runs:
        by_pair.setdefault(r["contestant"], []).append(r)

    rows, total_cost = [], 0.0
    for model, group in by_pair.items():
        total_cost += sum(r["cost_usd"] for r in group)
        clean = [r for r in group if not r["self_preference_risk"]]
        dirty = [r for r in group if r["self_preference_risk"]]
        row = {"contestant": model,
               "clean_judge": clean[0]["judge_model"] if clean else None,
               "same_vendor_judge": dirty[0]["judge_model"] if dirty else None,
               "judges_run": [r["judge_model"] for r in group],
               "cost_usd": round(sum(r["cost_usd"] for r in group), 4)}

        if clean:
            c = clean[0]
            row.update({"ours_wins": c["a_wins"], "theirs_wins": c["b_wins"],
                        "ties": c["ties"], "unparsed": c["unparsed"],
                        "n_decided": c["n_decided"],
                        "ours_win_rate": round(win_rate(c, "a"), 4),
                        "ours_wins_or_ties": round(
                            (c["a_wins"] + c["ties"]) / c["n_decided"], 4) if c["n_decided"] else None})
        if clean and dirty:
            # Positive delta = the contestant's own vendor rates it higher than a
            # neutral vendor does, i.e. self-preference in the direction expected.
            wc, wd = win_rate(clean[0], "b"), win_rate(dirty[0], "b")
            row["contestant_win_rate_clean_judge"] = None if wc is None else round(wc, 4)
            row["contestant_win_rate_same_vendor_judge"] = None if wd is None else round(wd, 4)
            row["self_preference_delta"] = (None if wc is None or wd is None
                                            else round(wd - wc, 4))
            row["agreement"] = agreement(clean[0], dirty[0])
        rows.append(row)

    rows.sort(key=lambda r: PANEL.index(r["contestant"]) if r["contestant"] in PANEL else 99)
    deltas = [r["self_preference_delta"] for r in rows if r.get("self_preference_delta") is not None]
    kappas = [r["agreement"]["kappa"] for r in rows
              if r.get("agreement") and r["agreement"]["kappa"] is not None]
    clean_beats = [r for r in rows if r.get("ours_win_rate") is not None and r["ours_win_rate"] > 0.5]
    modes = {r.get("mode", "preference") for r in runs}
    return {"ours": OURS, "split": SPLIT, "n_per_pair": protocol.JUDGE_SAMPLE_N,
            "mode": modes.pop() if len(modes) == 1 else sorted(modes),
            "pairs": rows,
            "pairs_won_on_clean_judge": f"{len(clean_beats)}/{len([r for r in rows if r.get('ours_win_rate') is not None])}",
            "mean_self_preference_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
            "mean_inter_judge_kappa": round(sum(kappas) / len(kappas), 4) if kappas else None,
            "total_cost_usd": round(total_cost, 4)}


MODE_BLURB = {
    "preference": ("Summaries only, no reference shown. This is the conventional "
                   "LLM-as-judge setup and it carries a verbosity bias: with no length "
                   "anchor, longer reads as more comprehensive. Median lengths against a "
                   "59-word gold reference are ours 56, GPT 81-86, Claude 190-223, so on "
                   "this panel that bias is loaded against us by construction."),
    "grounded": ("The judge is also shown the human reference answer and asked which "
                 "summary better matches it in content and emphasis. This is the mode "
                 "that speaks to the claim: the task is to reproduce what the annotator "
                 "wrote, and extra material the reference does not contain is not a merit."),
}


def compare_modes(pref_summary, grounded_summary):
    """Same pairs, same judges, same items. The only variable is whether the
    judge saw the reference, so the gap between the two is the verbosity effect."""
    g = {r["contestant"]: r for r in grounded_summary["pairs"]}
    rows = []
    for p in pref_summary["pairs"]:
        q = g.get(p["contestant"])
        if not q or p.get("ours_win_rate") is None or q.get("ours_win_rate") is None:
            continue
        rows.append({"contestant": p["contestant"],
                     "clean_judge": p["clean_judge"],
                     "ours_win_rate_preference": p["ours_win_rate"],
                     "ours_win_rate_grounded": q["ours_win_rate"],
                     "grounding_lift": round(q["ours_win_rate"] - p["ours_win_rate"], 4)})
    lifts = [r["grounding_lift"] for r in rows]
    return {"pairs": rows,
            "mean_grounding_lift": round(sum(lifts) / len(lifts), 4) if lifts else None}


def render_markdown(summary):
    mode = summary.get("mode", "preference")
    L = [f"# M4 blind judge panel ({mode}): {summary['ours']} vs frontier zero-shot",
         "",
         f"Split `{summary['split']}`, n={summary['n_per_pair']} per pair, blind pairwise, "
         "position-swapped on alternating items.",
         "Reported column is the CLEAN judge, the one whose vendor is not in the pair.",
         "",
         f"**Mode `{mode}`.** {MODE_BLURB.get(mode, '')}",
         "",
         "| Contestant | Clean judge | Ours | Theirs | Tie | Ours win rate | Unparsed |",
         "|---|---|---|---|---|---|---|"]
    for r in summary["pairs"]:
        if r.get("ours_win_rate") is None: continue
        L.append(f"| {r['contestant']} | {r['clean_judge']} | {r['ours_wins']} | "
                 f"{r['theirs_wins']} | {r['ties']} | {r['ours_win_rate']:.3f} | {r['unparsed']} |")
    L += ["", "## Self-preference, measured not assumed", "",
          "Same 60 items, same orientation, judged by the contestant's own vendor and by a "
          "neutral one. Positive delta = the contestant's vendor rates it higher than the "
          "neutral vendor does.", "",
          "| Contestant | Same-vendor judge | Theirs win rate (clean) | Theirs win rate (same vendor) | Delta | Agreement | Kappa |",
          "|---|---|---|---|---|---|---|"]
    for r in summary["pairs"]:
        if r.get("self_preference_delta") is None: continue
        ag = r["agreement"]
        L.append(f"| {r['contestant']} | {r['same_vendor_judge']} | "
                 f"{r['contestant_win_rate_clean_judge']:.3f} | "
                 f"{r['contestant_win_rate_same_vendor_judge']:.3f} | "
                 f"{r['self_preference_delta']:+.3f} | {ag['raw']:.3f} | {ag['kappa']} |")
    L += ["", f"- Pairs won on the clean judge: **{summary['pairs_won_on_clean_judge']}**",
          f"- Mean self-preference delta: **{summary['mean_self_preference_delta']}**",
          f"- Mean inter-judge kappa: **{summary['mean_inter_judge_kappa']}**",
          f"- Panel cost: **${summary['total_cost_usd']:.2f}**", ""]
    return "\n".join(L)


def render_mode_comparison(cmp_):
    L = ["## Does the verdict depend on seeing the reference?", "",
         "Identical pairs, judges and items; the only change is whether the judge was "
         "shown the human reference answer. A positive lift means grounding the judge "
         "moves the verdict toward us, i.e. the preference-mode result was partly "
         "measuring length.", "",
         "| Contestant | Clean judge | Ours win rate (preference) | Ours win rate (grounded) | Lift |",
         "|---|---|---|---|---|"]
    for r in cmp_["pairs"]:
        L.append(f"| {r['contestant']} | {r['clean_judge']} | "
                 f"{r['ours_win_rate_preference']:.3f} | {r['ours_win_rate_grounded']:.3f} | "
                 f"{r['grounding_lift']:+.3f} |")
    L += ["", f"- Mean grounding lift: **{cmp_['mean_grounding_lift']}**", ""]
    return "\n".join(L)


def main():
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--pairs", nargs="*", default=None, choices=PANEL)
    ap.add_argument("--judges", nargs="*", default=None, choices=JUDGE_PROVIDERS)
    ap.add_argument("--modes", nargs="*", default=None, choices=MODES)
    ap.add_argument("--n", type=int, default=protocol.JUDGE_SAMPLE_N)
    a = ap.parse_args()

    if a.cmd == "run":
        run_panel(pairs=a.pairs, providers=a.judges, modes=a.modes, n=a.n)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries, docs = {}, []
    for mode in (a.modes or MODES):
        runs = load_runs(pairs=a.pairs, providers=a.judges, mode=mode)
        if not runs:
            continue
        s = summarize(runs)
        summaries[mode] = s
        (OUT_DIR / f"panel-summary-{mode}.json").write_text(
            json.dumps(s, indent=2), encoding="utf-8")
        docs.append(render_markdown(s))
    if not summaries:
        print("no judge runs found; run the panel first"); return
    if "preference" in summaries and "grounded" in summaries:
        cmp_ = compare_modes(summaries["preference"], summaries["grounded"])
        (OUT_DIR / "panel-mode-comparison.json").write_text(
            json.dumps(cmp_, indent=2), encoding="utf-8")
        docs.append(render_mode_comparison(cmp_))
    md = "\n\n".join(docs)
    (OUT_DIR / "PANEL.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
