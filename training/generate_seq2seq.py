"""Predictions from a seq2seq checkpoint over located spans (pipeline front end)."""
import argparse, json, pathlib, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data.normalize import load_meetings, load_queries
from pipeline.run_pipeline import build_pipeline_prompt
from eval import protocol

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--locator", default="crossencoder", choices=["embed", "crossencoder"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.locator == "embed":
        from pipeline.locator_embed import select as locate
    else:
        from pipeline.locator_crossencoder import select as locate
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.checkpoint, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(a.split)}
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f, torch.no_grad():
        for q in load_queries(a.split)[:a.limit]:
            prompt = build_pipeline_prompt(q["query"], meetings[q["meeting_id"]], locate)
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=5120).to("cuda")
            gen = model.generate(**ids, max_new_tokens=protocol.MAX_NEW_TOKENS, num_beams=4)
            f.write(json.dumps({"query_id": q["query_id"],
                                "prediction": tok.decode(gen[0], skip_special_tokens=True).strip()}) + "\n")
            del ids, gen
            torch.cuda.empty_cache()
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
