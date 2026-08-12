"""Seq2seq fine-tune (DialogLED path). Full fine-tune, 460M fits 16 GB in bf16."""
import argparse, json, pathlib, sys, yaml, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval import protocol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    rows = [json.loads(l) for l in open(ROOT / cfg["train_file"], encoding="utf-8")]
    if a.dry_run: rows = rows[:4]
    def enc(e):
        x = tok(e["prompt"], truncation=True, max_length=cfg["max_source_len"])
        y = tok(text_target=e["response"], truncation=True, max_length=cfg["max_target_len"])
        x["labels"] = y["input_ids"]
        return x
    ds = Dataset.from_list(rows).map(enc, remove_columns=["query_id", "prompt", "response"])
    out_dir = ROOT / "checkpoints" / cfg["run_name"]
    trainer = Seq2SeqTrainer(
        model=model, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        args=Seq2SeqTrainingArguments(
            output_dir=str(out_dir), num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["micro_batch"],
            gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["lr"],
            bf16=True, logging_steps=10, save_strategy="epoch",
            max_steps=2 if a.dry_run else -1, seed=protocol.SEED, report_to=[]))
    trainer.train()
    model.save_pretrained(str(out_dir / "final")); tok.save_pretrained(str(out_dir / "final"))
    print(f"saved {out_dir / 'final'}")

if __name__ == "__main__":
    main()
