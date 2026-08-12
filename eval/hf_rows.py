"""Inference-only comparison rows: community QMSum checkpoints through the frozen scorer.

These seq2seq models were trained with 'query </s> transcript' style inputs and
short encoder budgets; we feed query-prefixed truncated transcripts, matching
their cards. Their rows chart the existing-specialist landscape; they are not
swept candidates.

Deviation from the brief's verbatim code, recorded here: the brief's
run_checkpoint loaded every checkpoint in torch.float16. On execution,
mikeadimech/pegasus-qmsum-meeting-summarization produces degenerate/empty
generations under fp16 (a known Pegasus fp16 numerical-instability issue,
confirmed by direct comparison: fp16 -> '' on multiple test queries, fp32/bf16
-> coherent summaries on the same inputs and inputs_ids). bart-large-cnn,
distilbart and led-base were confirmed non-degenerate under fp16 and kept as
specified. Pegasus alone is loaded in bfloat16 (same memory profile as fp16,
no precision-driven ROUGE effect observed in spot checks, avoids the collapse).
"""
import argparse, json, pathlib
from data.normalize import load_meetings, load_queries
from eval import protocol

ROOT = pathlib.Path(__file__).resolve().parents[1]

CHECKPOINTS = [
    {"repo": "mikeadimech/bart-large-cnn-qmsum-meeting-summarization", "slug": "bart-large-cnn", "word_budget": 700, "dtype": "float16"},
    {"repo": "mikeadimech/bart-qmsum-meeting-summarization", "slug": "distilbart", "word_budget": 700, "dtype": "float16"},
    {"repo": "mikeadimech/pegasus-qmsum-meeting-summarization", "slug": "pegasus", "word_budget": 700, "dtype": "bfloat16"},
    {"repo": "mikeadimech/longformer-qmsum-meeting-summarization", "slug": "led-base", "word_budget": 3000, "dtype": "float16"},
]

def build_input(query, utterances, word_budget):
    transcript = protocol.truncate_words(protocol.format_transcript(utterances), word_budget)
    return f"{query} </s> {transcript}"

def run_checkpoint(cp, split, limit=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    dtype = getattr(torch, cp.get("dtype", "float16"))
    tok = AutoTokenizer.from_pretrained(cp["repo"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cp["repo"], torch_dtype=dtype).to("cuda")
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
    out = ROOT / "results" / "preds" / f"row-{cp['slug']}-{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in load_queries(split)[:limit]:
            text = build_input(q["query"], meetings[q["meeting_id"]], cp["word_budget"])
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=tok.model_max_length).to("cuda")
            gen = model.generate(**ids, max_new_tokens=protocol.MAX_NEW_TOKENS, num_beams=4)
            f.write(json.dumps({"query_id": q["query_id"],
                                "prediction": tok.decode(gen[0], skip_special_tokens=True)}) + "\n")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, choices=[c["slug"] for c in CHECKPOINTS])
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cp = next(c for c in CHECKPOINTS if c["slug"] == a.slug)
    print(f"wrote {run_checkpoint(cp, a.split, a.limit)}")

if __name__ == "__main__":
    main()
