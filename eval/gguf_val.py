"""Gap 8: score the quantized GGUF export on validation.

The locate stage is byte-identical to the promoted pipeline (w375-l12
locator, 2,000-word budget, same `build_pipeline_prompt`); only the
summarizer differs: instead of the bf16 adapter under transformers, the
prompt goes to a llama.cpp server running the quantized base GGUF plus
the f16 LoRA GGUF, which is the artifact an on-device user runs.

Start the server first, then from the repo root:

  .venv\\Scripts\\python -m eval.gguf_val --out results/preds/gguf-q4km-val.jsonl
"""
import argparse
import json
import pathlib
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCATOR_CKPT = ROOT / "checkpoints" / "locator-crossencoder-w375-l12"
CHUNK_WORDS = 375
SPAN_BUDGET = 2000


def main():
    from data.normalize import load_meetings, load_queries
    from pipeline import locator_crossencoder as lc
    from pipeline.run_pipeline import build_pipeline_prompt
    from eval import protocol

    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8902")
    ap.add_argument("--split", default="val", choices=["val"],
                    help="val only: export fidelity is a decision input, and the "
                         "promoted system's single test touch stands")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    lc.CKPT, lc._MODEL = LOCATOR_CKPT, None
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(a.split)}
    queries = load_queries(a.split)[: a.limit]

    props = requests.get(f"{a.server}/props", timeout=30).json()
    model_path = props.get("model_path") or props.get("default_generation_settings", {}).get("model")
    print(f"server model: {model_path}", flush=True)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            prompt = build_pipeline_prompt(q["query"], meetings[q["meeting_id"]],
                                           lc.select, SPAN_BUDGET, CHUNK_WORDS)
            t1 = time.perf_counter()
            r = requests.post(f"{a.server}/completion", json={
                "prompt": prompt + " ",
                "n_predict": protocol.MAX_NEW_TOKENS,
                "temperature": 0,
                "top_k": 1,
                "cache_prompt": False,
            }, timeout=600)
            r.raise_for_status()
            d = r.json()
            t2 = time.perf_counter()
            f.write(json.dumps({
                "query_id": q["query_id"],
                "prediction": d["content"].strip(),
                "locate_s": round(t1 - t0, 4),
                "generate_s": round(t2 - t1, 4),
                "latency_s": round(t2 - t0, 4),
                "prompt_tokens": d.get("tokens_evaluated"),
                "new_tokens": d.get("tokens_predicted"),
                "warmup": i == 0,
                "decode_deviations": {
                    "chunk_words": CHUNK_WORDS,
                    "span_budget_words": SPAN_BUDGET,
                    "locator_ckpt": "checkpoints/locator-crossencoder-w375-l12",
                    "runtime": "llama.cpp b9351 /completion, temp 0, top_k 1",
                    "quant": "Q4_K_M base + f16 LoRA GGUF",
                },
            }) + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(queries)}", flush=True)
    print(f"wrote {out} ({len(queries)} predictions)")


if __name__ == "__main__":
    main()
