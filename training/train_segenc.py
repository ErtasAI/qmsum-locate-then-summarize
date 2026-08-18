"""Fine-tune the released SegEnc checkpoint on span-regime data.

THE QUESTION. Their ported checkpoint scores 0.3530 val on full transcripts and 0.2900 on our
2,000-word located spans, so it loses 6.3 ROUGE-1 when taken off its training regime. But it was
TRAINED on full transcripts, so spans are off-distribution for it, which is the same
train/inference length-mismatch confound that invalidated this project's own budget ablation. This
run removes the confound: train it for the span regime and see whether it recovers.

  recovers toward 0.35  -> the advantage was always the input regime, and a retrieval-regime FiD is
                           competitive, so architecture is not the differentiator
  stays near 0.29       -> Fusion-in-Decoder genuinely needs long input, and a decoder-only model
                           is better suited to retrieval. A real architectural finding.

Either answer is publishable, which is the property worth having before spending GPU time.

PRECISION. fp32 master weights with a bf16 autocast forward, rather than a pure bf16 model. The
memory probe measured a pure-bf16 configuration at 4.18 GB and flagged that bf16 Adam moments with
no fp32 master weights are numerically questionable over a real run; fp32 masters roughly double
the optimiser footprint to about 7 GB, which still fits comfortably. Paying that is cheaper than
debugging a silently degraded fine-tune, and this repo has already been bitten once by a
precision-dependent collapse (Pegasus under fp16).

MEMORY. Gradient checkpointing is not optional. The reported recipe sets max_num_chunks=8;
stride produces 15 chunks of 512 tokens. The stock checkpoint comparison separately retains
the authors' max_num_chunks=32 setting (63 effective chunks).

SELECTION DISCIPLINE. Checkpoints are saved per epoch and NOTHING is selected here. Selection runs
afterwards on val in the real located-spans regime via `eval/segenc.py --mode spans`, and only then
does one test touch happen. Training loss is logged for diagnosis, never for selection: the whole
point is that the span regime differs from the training distribution, so train loss is not a proxy
for the score we care about.
"""
import argparse
import json
import pathlib
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizerFast, get_linear_schedule_with_warmup, set_seed

from eval import protocol
from eval.segenc import BartForMultiConditionalGeneration, ChunkTokenizer
from training.segenc_recipe import evaluation_command

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAX_TARGET_LEN = 256   # their --val_max_target_length


class SpanDataset(Dataset):
    def __init__(self, path, tokenizer, chunker, limit=None):
        self.rows = [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
                     if l.strip()]
        if limit:
            self.rows = self.rows[:limit]
        self.tok = tokenizer
        self.chunker = chunker

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.chunker(r["source"], r["query"])
        labels = self.tok(r["target"], return_tensors="pt", max_length=MAX_TARGET_LEN,
                          truncation=True, padding="max_length")["input_ids"].squeeze(0)
        labels = labels.masked_fill(labels == self.tok.pad_token_id, -100)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": labels}


def collate(batch):
    """Micro-batch is 1 by design: chunk counts are uniform under pad=True, but stacking several
    15x512 examples would put activation memory back over the card."""
    if len(batch) != 1:
        raise ValueError("micro-batch must be 1; raise --grad-accum instead")
    b = batch[0]
    return ({"input_ids": b["input_ids"].unsqueeze(0),
             "attention_mask": b["attention_mask"].unsqueeze(0),
             "labels": b["labels"].unsqueeze(0)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/external/socratic-qmsum")
    ap.add_argument("--train-file", default="data/built/segenc-spans.train.jsonl")
    ap.add_argument("--out", default="checkpoints/segenc-spans")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--epoch-offset", type=int, default=0,
                    help="absolute epoch number of the loaded checkpoint; continuation uses 4")
    ap.add_argument("--lr", type=float, default=3e-5,
                    help="lower than a from-scratch BART fine-tune: this CONTINUES from an "
                         "already-QMSum-tuned checkpoint and only needs to move regime")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-num-chunks", type=int, default=8,
                    help="8 plus stride gives the reported 15 effective chunks")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--limit", type=int, help="smoke test on the first N examples")
    a = ap.parse_args()

    set_seed(protocol.SEED)
    tok = BartTokenizerFast.from_pretrained(a.base)
    chunker = ChunkTokenizer(tok, max_num_chunks=a.max_num_chunks, pad=True)

    # fp32 weights, bf16 autocast forward. See the PRECISION note above.
    model = BartForMultiConditionalGeneration.from_pretrained(a.base, dtype=torch.float32)
    model.to("cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    ds = SpanDataset(a.train_file, tok, chunker, a.limit)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate)
    steps_per_epoch = max(1, len(ds) // a.grad_accum)
    total_steps = steps_per_epoch * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * a.warmup_ratio), total_steps)

    out_root = pathlib.Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{len(ds)} examples, {a.epochs} epochs, grad_accum {a.grad_accum}, "
          f"{total_steps} optimizer steps, lr {a.lr}", flush=True)

    history_path = out_root / "history.json"
    history = (json.loads(history_path.read_text(encoding="utf-8"))
               if history_path.exists() else [])
    step = 0
    for local_epoch in range(1, a.epochs + 1):
        epoch = a.epoch_offset + local_epoch
        t0, running, n = time.perf_counter(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / a.grad_accum
            loss.backward()
            running += float(out.loss.detach())
            n += 1
            if (i + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"  epoch {epoch} step {step}/{total_steps} "
                          f"loss {running / max(n, 1):.4f}", flush=True)
        mean_loss = running / max(n, 1)
        elapsed = time.perf_counter() - t0
        ckpt = out_root / f"epoch{epoch}"
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        history.append({"epoch": epoch, "mean_train_loss": round(mean_loss, 4),
                        "seconds": round(elapsed, 1), "checkpoint": str(ckpt)})
        print(f"epoch {epoch}: loss {mean_loss:.4f}, {elapsed / 60:.1f} min -> {ckpt}", flush=True)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    print("\nNothing selected here by design. Next, on VAL in the located-spans regime:")
    for h in history[-a.epochs:]:
        print("  " + " ".join(evaluation_command(
            h["checkpoint"], h["epoch"], a.max_num_chunks)))
    print("Baselines to beat: the un-fine-tuned port at 0.2900 val on this regime, and our own "
          "promoted system at 0.3539. Pick on val, then ONE test touch.")


if __name__ == "__main__":
    main()
