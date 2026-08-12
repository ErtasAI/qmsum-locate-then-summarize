"""Dump locate-then-pack pipeline prompts to sft-format jsonl (no LM involved).

Decouples locating (sentence-transformers, frozen .venv) from generation, so
base models needing a newer transformers than the locator stack supports
(a second environment on transformers 5) can generate via generate_sft.
Prompts come from the same build_pipeline_prompt as run_pipeline, so
generate_sft over a dump is prompt-identical to run_pipeline on the same
split/locator/budget.
"""
import argparse, json, pathlib
from data.normalize import load_meetings, load_queries
from pipeline.run_pipeline import build_pipeline_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--locator", default="crossencoder", choices=["embed", "crossencoder"])
    ap.add_argument("--span-budget", type=int, default=None,
                    help="override protocol.SPAN_BUDGET_WORDS (protocol deviation; "
                         "record it in the run notes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.locator == "embed":
        from pipeline.locator_embed import select as locate
    else:
        from pipeline.locator_crossencoder import select as locate
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(a.split)}
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in load_queries(a.split)[:a.limit]:
            prompt = build_pipeline_prompt(q["query"], meetings[q["meeting_id"]],
                                           locate, a.span_budget)
            f.write(json.dumps({"query_id": q["query_id"], "prompt": prompt}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
