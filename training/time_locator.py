"""Gap 7: time the two locator trainings.

Re-runs both cross-encoder locator trainings with their exact original
settings (epochs 2, batch 32, seeded through eval.protocol like the
originals) and records wall clock. Checkpoints go to disposable
directories and are deleted afterward: the promoted artifacts at
checkpoints/locator-crossencoder* are never touched, since this measures
cost, and the numbers the paper quotes belong to the original weights.

Run from the repo root:  .venv\\Scripts\\python -m training.time_locator
"""
import json
import pathlib
import shutil
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "training-runs" / "locator-training-time-20260804.json"

RUNS = [
    {
        "name": "locator-l6-w900-protocol-exact",
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "train_file": ROOT / "data" / "built" / "locator.train.jsonl",
        "expected_pairs": 13_738,
    },
    {
        "name": "locator-l12-w375-promoted",
        "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "train_file": ROOT / "data" / "built" / "locator.train.w375.jsonl",
        "expected_pairs": 33_655,
    },
]


def gpu_mem_in_use_mib():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def main():
    from pipeline import locator_crossencoder as lc
    import torch

    records = []
    for spec in RUNS:
        pairs = sum(1 for _ in spec["train_file"].open(encoding="utf-8"))
        assert pairs == spec["expected_pairs"], (
            f"{spec['name']}: {pairs} pairs, expected {spec['expected_pairs']}")
        ckpt = ROOT / "checkpoints" / f"_timing-{spec['name']}"
        if ckpt.exists():
            shutil.rmtree(ckpt)
        mem_before = gpu_mem_in_use_mib()
        print(f"=== timing {spec['name']} ({pairs} pairs, "
              f"GPU in use at start: {mem_before} MiB) ===", flush=True)
        base_orig, ckpt_orig, model_orig = lc.BASE, lc.CKPT, lc._MODEL
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            lc.BASE, lc.CKPT, lc._MODEL = spec["model"], ckpt, None
            lc.train(train_file=spec["train_file"])  # epochs=2, batch=32 defaults
        finally:
            lc.BASE, lc.CKPT, lc._MODEL = base_orig, ckpt_orig, model_orig
        elapsed = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        records.append({
            "run": spec["name"],
            "base_model": spec["model"],
            "train_pairs": pairs,
            "epochs": 2,
            "batch_size": 32,
            "wall_clock_s": round(elapsed, 1),
            "wall_clock_min": round(elapsed / 60, 2),
            "peak_allocated_gb": round(peak_gb, 3),
            "gpu_mem_in_use_at_start_mib": mem_before,
            "note": "timed re-run with original settings to a disposable directory; "
                    "the promoted checkpoint was not touched",
        })
        print(f"    {elapsed/60:.2f} min, peak {peak_gb:.2f} GB", flush=True)
        shutil.rmtree(ckpt, ignore_errors=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
