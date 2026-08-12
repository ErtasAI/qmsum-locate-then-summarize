"""Generate predictions from a QLoRA adapter over an SFT-format dataset."""
import argparse, json, pathlib, torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from eval import protocol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True, help="sft-format jsonl with query_id + prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.adapter)
    model = AutoPeftModelForCausalLM.from_pretrained(a.adapter, device_map="cuda",
                                                     torch_dtype=torch.bfloat16)
    model.eval()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")][:a.limit]
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f, torch.no_grad():
        for r in rows:
            ids = tok(r["prompt"] + " ", return_tensors="pt").to("cuda")
            gen = model.generate(**ids, max_new_tokens=protocol.MAX_NEW_TOKENS, do_sample=False)
            text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({"query_id": r["query_id"], "prediction": text.strip()}) + "\n")
            del ids, gen
            torch.cuda.empty_cache()
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
