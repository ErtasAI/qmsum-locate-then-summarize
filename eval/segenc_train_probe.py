"""Can SegEnc be fine-tuned on located spans inside 16 GB? Measure before committing.

Third time this pattern has paid for itself today: a beam-search config that needed 27.3 GB, an
inference config that needed 15.91 GB of a 16.3 GB card, and now a training config. Probing costs
minutes; discovering it after building a training loop costs hours.

The question is task 6's: fine-tune the released SegEnc checkpoint on OUR located spans and see
whether it recovers from 0.2900 toward its full-transcript 0.3530. That separates "their advantage
is the input regime" from "their advantage is being trained for the input regime".

Two axes matter for memory and one of them is an experimental-design choice, not just an
engineering one:

* **Chunk count.** The ablation's spans arm used pad=True, so 63 chunks of which most are padding.
  Fine-tuning at 63 keeps the fine-tuned model directly comparable to that 0.2900 baseline. Fine-
  tuning at a chunk count matched to the span length (2,000 words is about 2,700 tokens, so roughly
  6 chunks) is what a practitioner would actually do and is far cheaper, but it changes two things
  at once. Both are probed so the choice can be made on cost with the tradeoff visible.
* **Adaptation.** Full fine-tuning of 406M fp32 params costs weights + grads + two Adam moments,
  roughly 6.4 GB before any activation, which is most of the card. LoRA removes the optimiser state
  almost entirely and matches what the rest of this project does.

Reports peak allocated memory for a real forward AND backward with labels, because a
forward-only measurement is the mistake that would make this probe useless.
"""
import argparse
import json
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]

CONFIGS = [
    # (label, n_chunks, lora, grad_checkpointing)
    ("63ch full-ft  no-ckpt", 63, False, False),
    ("63ch full-ft  ckpt",    63, False, True),
    ("63ch lora     ckpt",    63, True,  True),
    ("16ch lora     ckpt",    16, True,  True),
    ("8ch  lora     ckpt",     8, True,  True),
    ("8ch  full-ft  ckpt",     8, False, True),
]


def build_batch(tok, n_chunks, device, target_words=70):
    """One training example shaped (1, n_chunks, 512) with labels."""
    from eval.segenc import CHUNK_SIZE
    ids = torch.randint(4, 50000, (1, n_chunks, CHUNK_SIZE), device=device)
    mask = torch.ones_like(ids)
    target = " ".join(f"word{i}" for i in range(target_words))
    labels = tok(target, return_tensors="pt", max_length=256, truncation=True,
                 padding="max_length")["input_ids"].to(device)
    labels[labels == tok.pad_token_id] = -100
    return ids, mask, labels


def probe(model_dir, config, tok):
    from eval.segenc import BartForMultiConditionalGeneration
    label, n_chunks, use_lora, ckpt = config
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = BartForMultiConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.to("cuda")
    if ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        # Required whenever gradient checkpointing is combined with a frozen base and adapters.
        # Without it the checkpointed segment sees no input requiring grad, the backward has no
        # path to the LoRA weights, and `loss.backward()` raises "element 0 of tensors does not
        # require grad". The first run of this probe hit exactly that and reported it as though
        # LoRA did not fit, which it does, comfortably.
        model.enable_input_require_grads()
    if use_lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_2_SEQ_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    ids, mask, labels = build_batch(tok, n_chunks, "cuda")
    err = None
    try:
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        loss = float(out.loss.detach())
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        loss, err = None, str(exc).split("\n")[0][:110]
    peak = torch.cuda.max_memory_allocated() / 1e9
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model, opt, ids, mask, labels
    torch.cuda.empty_cache()
    return {"config": label, "n_chunks": n_chunks, "lora": use_lora, "grad_ckpt": ckpt,
            "peak_gb": round(peak, 2), "trainable_params": trainable,
            "loss": round(loss, 4) if loss is not None else None, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="results/external/socratic-qmsum")
    a = ap.parse_args()
    from transformers import BartTokenizerFast
    tok = BartTokenizerFast.from_pretrained(a.model_dir)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    # Headroom, not the raw total: this platform silently spills to host memory instead of
    # raising, so a run that completes is NOT evidence that it fit. Beam search "completed" at
    # 27.3 GB on this card and took 431 seconds a query. Anything within 15% of the total is
    # treated as not fitting.
    ceiling = total * 0.85
    print(f"card: {torch.cuda.get_device_name(0)}, {total:.2f} GB "
          f"(treating {ceiling:.2f} GB as the practical ceiling)\n")
    print(f"{'config':24} {'peak':>8} {'trainable':>12}  {'fits':>5}  loss")
    rows = []
    for cfg in CONFIGS:
        r = probe(a.model_dir, cfg, tok)
        # A completed step above the ceiling means it spilled, not that it fit.
        r["fits"] = bool(r["error"] is None and r["peak_gb"] <= ceiling)
        r["spilled"] = bool(r["error"] is None and r["peak_gb"] > ceiling)
        rows.append(r)
        fits = "yes" if r["fits"] else ("SPILL" if r["spilled"] else "no")
        note = f"  ERR {r['error']}" if r["error"] else f"  loss {r['loss']}"
        print(f"{r['config']:24} {r['peak_gb']:6.2f}GB {r['trainable_params']:12,}  {fits:>5}{note}",
              flush=True)
    out = ROOT / "results" / "segenc-train-probe.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    ok = [r for r in rows if r["fits"]]
    if ok:
        best = max(ok, key=lambda r: r["n_chunks"])
        print(f"\nhighest chunk count that genuinely FITS: {best['config']} at {best['peak_gb']}GB")
        print("Note: bf16 params with bf16 Adam moments and no fp32 master weights. A real run "
              "should use fp32 master weights or fp32 optimiser state, which roughly doubles the "
              "optimiser footprint; budget for that before scheduling.")
    else:
        print("\nNOTHING FITS. Span-regime fine-tuning of SegEnc is not feasible on this card.")


if __name__ == "__main__":
    main()
