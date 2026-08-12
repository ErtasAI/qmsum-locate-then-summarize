"""Peak training VRAM as a function of sequence length, for the promoted configuration.

WHY THIS EXISTS. The paper's efficiency table has said "not instrumented, < 16 GB,
trains" for the training row since the table was written. On 2026-07-31 two attempts to
retrain the promoted configuration for a seed sweep died with CUDA out-of-memory in the
backward pass, on the same card that trained it originally. So "< 16 GB" is true only in
the sense that the run fits when nothing else is on the card, and the blank cell was
hiding the fact that the configuration sits at the hardware limit rather than under it.

This measures the actual requirement instead of asserting a bound: peak allocated and
peak reserved for one real training step, at each of several real sequence lengths drawn
from the training set, plus the resident-weights floor.

WHAT IT REPLICATES. The same 4-bit base, the same LoRA config, gradient checkpointing,
the same 8-bit optimizer, and the same tokenization and padding, by importing `encode`
and `Collator` from `train_lora` rather than reimplementing them. One optimizer step is
taken before measuring so that optimizer state is resident, which is where a naive probe
understates the requirement.

WHAT IT DOES NOT REPLICATE. `Trainer`'s accumulation loop. With `micro_batch=1` the peak
is set by a single example's forward and backward, and gradient accumulation adds only
the LoRA gradient buffers (11M parameters), which are resident throughout either way. The
figure is therefore the per-step peak, and the run's true peak is that plus whatever
`Trainer` holds around it. Reported as a floor for the run, not as a ceiling.

Reads the same convention as `eval/vram.py`: peak ALLOCATED leads, peak RESERVED is
recorded beside it, weights are inside the number.
"""
import argparse, json, pathlib, sys, yaml, torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval import vram
from training.train_lora import encode, Collator, gpu_memory_used_mib


def token_lengths(rows, tok, max_len):
    return [len(encode(r, tok, max_len)["input_ids"]) for r in rows]


def pick_probe_rows(rows, lengths, n_points):
    """Real training examples spanning the length distribution, longest always included.

    The longest example is what decides whether the run survives, so it is never sampled
    away. The rest are evenly spaced by RANK rather than by length, because the length
    distribution is what it is and quoting a curve over lengths that do not occur in the
    data would describe a configuration nobody trains.
    """
    order = sorted(range(len(rows)), key=lambda i: lengths[i])
    if n_points <= 1:
        return [order[-1]]
    if n_points >= len(order):
        picks = order
    else:
        step = (len(order) - 1) / (n_points - 1)
        picks = sorted({order[min(len(order) - 1, round(k * step))] for k in range(n_points)})
    if order[-1] not in picks:
        picks.append(order[-1])
    return sorted(picks, key=lambda i: lengths[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--points", type=int, default=8, help="how many lengths to probe")
    ap.add_argument("--out", default="results/training-runs/vram-curve.json")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    used_at_start = gpu_memory_used_mib()
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(ROOT / cfg["train_file"], encoding="utf-8")]
    lengths = token_lengths(rows, tok, cfg["max_seq_len"])
    picks = pick_probe_rows(rows, lengths, a.points)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_quant_type="nf4"))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
        target_modules="all-linear", task_type="CAUSAL_LM"))
    model.train()

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    collate = Collator(tok.pad_token_id)

    torch.cuda.synchronize()
    weights_floor = {"peak_vram_gb": vram.peak_gb(), "peak_vram_reserved_gb": vram.peak_reserved_gb()}
    print(f"resident floor after load: {weights_floor}")

    points, failed_at = [], None
    for idx in picks:
        n_tok = lengths[idx]
        batch = collate([encode(rows[idx], tok, cfg["max_seq_len"])])
        batch = {k: v.to("cuda") for k, v in batch.items()}
        # Return cached blocks to the driver before each point. Without this the RESERVED
        # column measures the probe's own allocation history rather than the length being
        # probed: it climbed monotonically to 21.0 GB on a 17.1 GB card in the first run
        # of this script, and then reported an OOM at the longest example that does not
        # occur when that example is run in a fresh process. ALLOCATED was unaffected
        # either way, since it counts live tensor bytes.
        torch.cuda.empty_cache()
        vram.reset()
        try:
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        except torch.OutOfMemoryError as e:
            failed_at = {"tokens": n_tok, "error": type(e).__name__}
            print(f"OOM at {n_tok} tokens")
            break
        except RuntimeError as e:
            # cuBLAS reports exhaustion as an execution failure rather than an OOM.
            failed_at = {"tokens": n_tok, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"runtime failure at {n_tok} tokens: {str(e)[:120]}")
            break
        rec = {"tokens": n_tok, "words": len(rows[idx]["prompt"].split()) + len(rows[idx]["response"].split()),
               "peak_vram_gb": vram.peak_gb(), "peak_vram_reserved_gb": vram.peak_reserved_gb()}
        points.append(rec)
        print(f"  {n_tok:6d} tok -> {rec['peak_vram_gb']:6.3f} GB allocated / {rec['peak_vram_reserved_gb']:6.3f} reserved")

    total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3)
    out = {
        "config": str(pathlib.Path(a.config).as_posix()),
        "base_model": cfg["base_model"],
        "max_seq_len": cfg["max_seq_len"],
        "n_train_examples": len(rows),
        "token_length_min": min(lengths), "token_length_median": sorted(lengths)[len(lengths) // 2],
        "token_length_max": max(lengths),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_gb": total_gb,
        "gpu_memory_used_mib_at_start": used_at_start,
        "resident_weights_floor": weights_floor,
        "points": points,
        "failed_at": failed_at,
        "note": ("Per-step peak for one example at each length, optimizer state resident. "
                 "Floor for a full run, not a ceiling: Trainer holds additional state around this."),
    }
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
