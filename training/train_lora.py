"""QLoRA SFT for decoder LMs. Loss on response tokens only. 16 GB VRAM budget.

Instrumented 2026-07-31 to close the last unmeasured cells in the paper's efficiency table
(gaps 2, 4, 6 and 7) and to make seed replication possible without touching the frozen
protocol. Every run now writes `results/training-runs/<run_name>.json`.

Four conventions are fixed here.

**Peak VRAM is one window over the whole run, not per step.** `eval/vram.py` resets per
query at inference because memory there tracks input length and a median hides the query
that sizes the card. Training has no such distribution to report: the question is whether
the configuration fits at all, so the honest figure is the maximum over the entire run
with weights, optimizer state and gradients all resident. Allocated leads and reserved is
recorded beside it, same as inference, so the two tables are read the same way.

**GPU memory in use before the run starts is recorded.** The v8 wall-clock of 4h 43m was
measured against an unrelated job sharing the card and nobody noticed until the step
times were read back months later. A nonzero `gpu_memory_used_mib_at_start` is the flag
that a wall-clock from this run is not a clean one, recorded at the moment it is knowable
rather than reconstructed from a log afterwards.

**A seed that differs from the frozen protocol suffixes the run name.** `eval/protocol.py`
is frozen by decision and `tests/test_protocol_frozen.py` enforces it, so the seed arrives
as a CLI argument defaulting to `protocol.SEED`. Auto-suffixing means a seed sweep cannot
silently overwrite the promoted checkpoint by inheriting its `run_name` from the config.

**An existing non-empty output directory is refused.** `checkpoints/m3-fusion-lfm2.5-1.2b`
is the paper's headline system and is not reproducible from anything else in the repo.
`--force` is available and deliberate.
"""
import argparse, json, os, pathlib, shutil, subprocess, sys, time, yaml, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          Trainer, TrainerCallback, TrainingArguments)
from peft import LoraConfig, get_peft_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval import protocol, vram

def encode(example, tok, max_len):
    prompt_ids = tok(example["prompt"] + " ", add_special_tokens=True)["input_ids"]
    resp_ids = tok(example["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + resp_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + resp_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}

class Collator:
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        return {"input_ids": torch.tensor([b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]),
                "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
                "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch])}

def run_name_for(cfg_run_name, seed, dry_run=False):
    """Config name at the frozen seed; `-seedN` otherwise; `_dryrun-` prefixed for a smoke test.

    The suffix is what stops a seed sweep from writing over the promoted checkpoint,
    so it is derived rather than left to whoever invokes the script.

    Dry runs get their own disposable directory. Exempting them from the overwrite guard
    instead would be worse than having no guard: `--dry-run` on the promoted config would
    sail past the refusal and then `save_pretrained` a two-step adapter over the paper's
    headline system, which is exactly the loss the guard exists to prevent.
    """
    name = cfg_run_name if seed == protocol.SEED else f"{cfg_run_name}-seed{seed}"
    return f"_dryrun-{name}" if dry_run else name

def param_counts(model):
    """Trainable and total parameters. Closes efficiency-table gap 6.

    `print_trainable_parameters()` prints and returns None, so the numbers are recomputed
    here to land in the artifact. Counting the 4-bit base is subtle: bitsandbytes stores
    `Params4bit` with two values packed per byte, so `numel()` reads half the logical
    parameter count. `element_size() == 1` identifies those tensors and the count is
    doubled, which is why `total` comes out near 1.2B rather than near 700M.
    """
    trainable = total = 0
    for p in model.parameters():
        n = p.numel()
        if getattr(p, "quant_state", None) is not None or (p.dtype == torch.uint8 and p.element_size() == 1):
            n *= 2
        total += n
        if p.requires_grad:
            trainable += p.numel()
    return {"trainable_params": trainable, "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4) if total else None}

def dir_bytes(path):
    """Total size on disk of a saved adapter, and the largest single file in it.

    Both numbers are reported because they answer different questions. `save_pretrained`
    writes the tokenizer alongside the weights, so the directory total is what a release
    tarball weighs, while the largest file is the adapter tensor itself, which is the
    figure to quote against the base model's size.
    """
    files = [f for f in path.rglob("*") if f.is_file()]
    if not files:
        return None
    largest = max(files, key=lambda f: f.stat().st_size)
    return {"bytes": sum(f.stat().st_size for f in files),
            "mb": round(sum(f.stat().st_size for f in files) / 1e6, 2),
            "n_files": len(files),
            "largest_file": largest.name,
            "largest_file_mb": round(largest.stat().st_size / 1e6, 2)}

class EmptyCacheCallback(TrainerCallback):
    """Return cached CUDA blocks to the driver every N optimizer steps.

    Diagnosed 2026-07-31. A single training step of this configuration fits comfortably:
    the longest example in the set (4,481 tokens) peaks at 11.89 GB allocated and 13.38 GB
    reserved on a 17.09 GB card, leaving 3.7 GB. A full RUN of it nonetheless died twice
    with the card reporting about 16.7 GB in use, because the caching allocator's
    reservation grows across steps as blocks of many different shapes accumulate, and
    6% of the training examples are long enough to demand large contiguous blocks.

    `torch.cuda.empty_cache()` changes no numerics whatsoever. It costs a little wall
    clock, which is why it is opt-in and recorded in the run artifact: a timing figure
    from a run with this enabled is not comparable to one without it.
    """
    def __init__(self, every): self.every = every
    def on_step_end(self, args, state, control, **kw):
        if self.every and state.global_step % self.every == 0:
            torch.cuda.empty_cache()


def gpu_memory_used_mib():
    """Memory already in use on GPU 0, or None. A nonzero value means co-tenancy."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15, check=True).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="4 examples, 2 steps, verifies the loop")
    ap.add_argument("--seed", type=int, default=protocol.SEED,
                    help=f"training seed; default {protocol.SEED} (protocol). Any other value suffixes the run name.")
    ap.add_argument("--force", action="store_true", help="allow writing into a non-empty checkpoint directory")
    ap.add_argument("--empty-cache-every", type=int, default=0,
                    help="release cached CUDA blocks every N optimizer steps (0 = off, the original "
                         "training path). Changes no numerics; costs some wall clock. Use on a card "
                         "that is not exclusively ours. Recorded in the run artifact.")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    run_name = run_name_for(cfg["run_name"], a.seed, a.dry_run)
    out_dir = ROOT / "checkpoints" / run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not a.force:
        sys.exit(f"refusing to write into non-empty {out_dir}\n"
                 f"pass --force if overwriting is intended, or use a different --seed")

    gpu_used_at_start = gpu_memory_used_mib()
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(ROOT / cfg["train_file"], encoding="utf-8")]
    if a.dry_run: rows = rows[:4]
    ds = Dataset.from_list(rows).map(lambda e: encode(e, tok, cfg["max_seq_len"]),
                                     remove_columns=["query_id", "prompt", "response"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_quant_type="nf4"))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
        target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    counts = param_counts(model)
    trainer = Trainer(model=model, train_dataset=ds, data_collator=Collator(tok.pad_token_id),
                      args=TrainingArguments(
                          output_dir=str(out_dir), num_train_epochs=cfg["epochs"],
                          per_device_train_batch_size=cfg["micro_batch"],
                          gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["lr"],
                          lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
                          logging_steps=10, save_strategy="epoch", optim="adamw_bnb_8bit",
                          max_steps=2 if a.dry_run else -1, seed=a.seed, report_to=[]))
    if a.empty_cache_every:
        trainer.add_callback(EmptyCacheCallback(a.empty_cache_every))
    vram.reset()
    t0 = time.perf_counter()
    result = trainer.train()
    wall_s = time.perf_counter() - t0
    peak = {"peak_vram_gb": vram.peak_gb(), "peak_vram_reserved_gb": vram.peak_reserved_gb()}

    final = out_dir / "final"
    model.save_pretrained(str(final)); tok.save_pretrained(str(final))
    record = {
        "run_name": run_name,
        "config": str(pathlib.Path(a.config).as_posix()),
        "base_model": cfg["base_model"],
        "seed": a.seed,
        "seed_is_protocol": a.seed == protocol.SEED,
        "dry_run": a.dry_run,
        "epochs": cfg["epochs"],
        "n_train_examples": len(rows),
        "max_seq_len": cfg["max_seq_len"],
        "lora": cfg["lora"],
        "optimizer_steps": int(result.global_step),
        "train_loss": round(float(result.training_loss), 6),
        "wall_clock_s": round(wall_s, 2),
        "wall_clock_hms": time.strftime("%Hh %Mm %Ss", time.gmtime(wall_s)),
        "s_per_step": round(wall_s / result.global_step, 3) if result.global_step else None,
        **counts,
        **peak,
        "adapter_on_disk": dir_bytes(final),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_used_mib_at_start": gpu_used_at_start,
        "contended": None if gpu_used_at_start is None else gpu_used_at_start > 500,
        "torch_version": torch.__version__,
        # Recorded because it changes wall clock without changing numerics. The allocator
        # reserves far more than it allocates on this configuration (13.1 GB reserved
        # against 5.4 GB allocated on a two-step probe), which leaves nothing on a 16 GB
        # card for the desktop and drove a hard OOM on 2026-07-31. Any timing compared
        # across runs has to know whether this was set.
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "empty_cache_every": a.empty_cache_every or None,
    }
    runs_dir = ROOT / "results" / "training-runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{run_name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"saved {final}")

if __name__ == "__main__":
    main()
