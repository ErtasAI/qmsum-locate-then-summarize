"""Find a beam-search config that fits a 16 GB card on this task's longest prompts.

The first decode-sweep attempt ran at a median 6.5s per query with a 20x cliff above roughly
4,000 prompt tokens: 3,880 tokens took 10.7s, 4,051 took 239s, worst case 338s. GPU
utilisation sat at 44% with 15.9 of 16.3 GB allocated, so the cliff is host-memory spill
rather than compute. Correlation of generate time with prompt length was 0.784 and with output
length 0.193, which points at prefill rather than decode.

Cause: HuggingFace beam search expands `input_ids` to batch times beams BEFORE the prefill
forward, so a 4,300-token prompt is encoded four times over. Peak memory therefore scales with
prompt length times beams, and this task's prompts are long by construction.

This probe measures peak allocated memory and wall time for candidate configs on the LONGEST
prompts in the split rather than on the first few queries, which is what made the original
smoke test look fine. It writes nothing except its own report; it never touches a preds file.
"""
import argparse
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

CANDIDATES = [
    {"name": "greedy",            "kw": {}},
    {"name": "b2",                "kw": {"num_beams": 2}},
    {"name": "b4",                "kw": {"num_beams": 4}},
    {"name": "b4-lowmem",         "kw": {"num_beams": 4, "low_memory": True}},
    {"name": "b2-lowmem",         "kw": {"num_beams": 2, "low_memory": True}},
]


def main():
    import torch
    from transformers import AutoTokenizer
    from peft import AutoPeftModelForCausalLM
    from data.normalize import load_meetings, load_queries
    from pipeline.chunker import windows
    from pipeline.locator_crossencoder import select as locate
    from pipeline.run_pipeline import build_pipeline_prompt
    from pipeline import lfm2_beam_fix
    from eval import protocol

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=3, help="how many of the longest prompts to test")
    a = ap.parse_args()

    lfm2_beam_fix.apply()
    adapter = ROOT / "checkpoints" / "m3-fusion-lfm2.5-1.2b" / "final"
    tok = AutoTokenizer.from_pretrained(adapter)
    model = AutoPeftModelForCausalLM.from_pretrained(adapter, device_map="cuda",
                                                     dtype=torch.bfloat16)
    model.eval()

    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(a.split)}
    print("building prompts to find the longest ones...", flush=True)
    prompts = []
    for q in load_queries(a.split):
        p = build_pipeline_prompt(q["query"], meetings[q["meeting_id"]], locate)
        prompts.append((len(tok(p)["input_ids"]), q["query_id"], p))
    prompts.sort(reverse=True)
    worst = prompts[:a.n]
    print("longest prompts:", [(n, qid) for n, qid, _ in worst], flush=True)

    results = []
    with torch.no_grad():
        for cand in CANDIDATES:
            for n_tok, qid, prompt in worst:
                ids = tok(prompt + " ", return_tensors="pt").to("cuda")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    gen = model.generate(**ids, max_new_tokens=protocol.MAX_NEW_TOKENS,
                                         do_sample=False, **cand["kw"])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    new_tokens = int(gen.shape[1] - ids["input_ids"].shape[1])
                    err = None
                    del gen
                except RuntimeError as exc:  # OOM and friends
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    new_tokens, err = None, str(exc)[:120]
                peak = torch.cuda.max_memory_allocated() / 1e9
                row = {"config": cand["name"], "query_id": qid, "prompt_tokens": n_tok,
                       "seconds": round(elapsed, 2), "peak_gb": round(peak, 2),
                       "new_tokens": new_tokens, "error": err}
                results.append(row)
                print(f"  {row['config']:12} {n_tok:5}tok  {row['seconds']:8.2f}s  "
                      f"peak {row['peak_gb']:5.2f}GB  new {new_tokens}"
                      + (f"  ERR {err}" if err else ""), flush=True)
                del ids
                torch.cuda.empty_cache()

    out = ROOT / "results" / "beam-memory-probe.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    print("\nper-config worst case across the tested prompts:")
    for cand in CANDIDATES:
        rows = [r for r in results if r["config"] == cand["name"]]
        if not rows:
            continue
        print(f"  {cand['name']:12} max {max(r['seconds'] for r in rows):8.2f}s  "
              f"peak {max(r['peak_gb'] for r in rows):5.2f}GB  "
              f"errors {sum(1 for r in rows if r['error'])}")


if __name__ == "__main__":
    main()
