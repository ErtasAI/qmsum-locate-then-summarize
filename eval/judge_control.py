"""Control experiment: does the judge prefer the gold reference itself?

WHY. The panel asks a judge which of two summaries is better. Before any of its
verdicts can be read as evidence about quality, the instrument needs calibrating,
and there is one control that calibrates it exactly: enter the HUMAN REFERENCE
ANSWER as a contestant.

In `grounded` mode this is a tautology by construction. The judge is shown the
reference and asked which summary better matches it, and one of the two summaries
IS that reference, verbatim. A judge measuring what the prompt asks must pick it
at close to 100%. Anything materially below that is not a close call, it is the
instrument failing: the judge is scoring prose quality, or length, or fluency,
and reporting it as reference-matching.

In `preference` mode the same pairing measures something different and also worth
knowing: whether a reference-free judge prefers a 59-word human-written answer to
a 200-word model-written one. QMSum's references are terse and written by human
annotators. If the judge rejects them, then reference-free preference judging is
not a valid proxy for this task, whoever it is applied to, and that conclusion is
independent of how our own system scores.

Either way the finding is about the measuring instrument rather than about us,
which is what makes it worth the $2 it costs.

    python -m eval.judge_control --against gpt-5.6-luna
    python -m eval.judge_control --against gpt-5.6-luna --modes grounded
"""
import argparse, itertools, json, pathlib

from data.normalize import load_queries
from eval import judge as J
from eval import protocol
from eval import run_judge_panel as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTROL_DIR = ROOT / "results" / "judge_controls"

# Deliberately NOT under results/preds/: this is a control fixture, not a system
# row, and it must never be picked up by the scoreboard or the chart builder.
REFERENCE_AS_SYSTEM = CONTROL_DIR / "reference-as-system-{split}.jsonl"


def build_reference_preds(split=P.SPLIT, path=None):
    """A predictions file whose every prediction is the gold reference.

    `model_snapshot` is omitted on purpose: the references are human-written, so
    they belong to no vendor and must never trip the self-preference check.
    """
    path = pathlib.Path(str(path or REFERENCE_AS_SYSTEM).format(split=split))
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"query_id": q["query_id"], "prediction": q["reference"]}
            for q in load_queries(split)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def run_control(against, providers=None, modes=None, split=P.SPLIT, n=None,
                clients=None, quiet=False):
    providers = providers or P.JUDGE_PROVIDERS
    modes = modes or P.MODES
    n = n or protocol.JUDGE_SAMPLE_N
    clients = clients or {}
    ref_preds = build_reference_preds(split)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for mode, provider in itertools.product(modes, providers):
        judge_model = J.DEFAULT_JUDGE_MODEL[provider]
        tag = f"reference-vs-{against}__judge-{judge_model}__{mode}"
        if provider not in clients:
            clients[provider] = P.make_client(provider)
        if not quiet:
            print(f"[control] {mode}: GOLD REFERENCE vs {against}  <-  {judge_model} (n={n})",
                  flush=True)
        tally = J.run_judgement(
            ref_preds, P.theirs_preds(against, split), split, n,
            clients[provider], provider=provider, model=judge_model,
            cache_dir=CONTROL_DIR / "cache" / tag, mode=mode)
        tally["contestant"] = against
        tally["ours"] = "GOLD-REFERENCE"
        tally["split"] = split
        tally["is_control"] = True
        (CONTROL_DIR / f"{tag}.json").write_text(json.dumps(tally, indent=2), encoding="utf-8")
        if not quiet:
            print(f"          reference {tally['a_wins']} / {against} {tally['b_wins']} / "
                  f"tie {tally['ties']}  ${tally['cost_usd']:.2f}", flush=True)
        runs.append(tally)
    return runs


def summarize_control(runs):
    rows = []
    for r in runs:
        d = r["n_decided"]
        rows.append({"contestant": r["contestant"], "mode": r.get("mode", "preference"),
                     "judge_model": r["judge_model"],
                     "reference_wins": r["a_wins"], "contestant_wins": r["b_wins"],
                     "ties": r["ties"], "n_decided": d,
                     "reference_win_rate": round(r["a_wins"] / d, 4) if d else None})
    return {"control": "gold reference entered as a contestant", "runs": rows}


def render_markdown(summary):
    L = ["### Control: the gold reference as a contestant", "",
         "The human reference answer is entered as one of the two summaries. In "
         "`grounded` mode the judge is also shown that same reference and asked which "
         "summary matches it better, so a judge measuring what the prompt asks must "
         "pick it at close to 1.000. A low rate does not mean the reference is bad, it "
         "means the judge is not measuring reference-matching.", "",
         "| Mode | Judge | Contestant | Reference wins | Contestant wins | Reference win rate |",
         "|---|---|---|---|---|---|"]
    for r in summary["runs"]:
        rate = "n/a" if r["reference_win_rate"] is None else f"{r['reference_win_rate']:.3f}"
        L.append(f"| {r['mode']} | {r['judge_model']} | {r['contestant']} | "
                 f"{r['reference_wins']} | {r['contestant_wins']} | {rate} |")
    L.append("")
    return "\n".join(L)


def main():
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--against", default="gpt-5.6-luna", choices=P.PANEL)
    ap.add_argument("--judges", nargs="*", default=None, choices=P.JUDGE_PROVIDERS)
    ap.add_argument("--modes", nargs="*", default=None, choices=P.MODES)
    ap.add_argument("--n", type=int, default=protocol.JUDGE_SAMPLE_N)
    a = ap.parse_args()
    runs = run_control(a.against, providers=a.judges, modes=a.modes, n=a.n)
    summary = summarize_control(runs)
    (CONTROL_DIR / "control-summary.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8")
    md = render_markdown(summary)
    (CONTROL_DIR / "CONTROL.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
