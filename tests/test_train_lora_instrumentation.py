"""Contract tests for the training-run instrumentation added 2026-07-31.

These run on CPU and import no model. They pin the four things that, if they drift,
produce either a wrong number in the paper's efficiency table or a destroyed checkpoint:
the seed-to-run-name mapping, the 4-bit parameter count, the adapter footprint, and the
refusal to overwrite an existing checkpoint directory.

The overwrite guard is the one worth stating plainly. `checkpoints/m3-fusion-lfm2.5-1.2b`
holds the paper's headline system, it is not reproducible from anything else in the repo,
and the natural way to run a seed sweep (reuse the config, change the seed) would have
written straight over it before this change.
"""
import json, pathlib, subprocess, sys, types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import protocol
from training.train_lora import run_name_for, param_counts, dir_bytes


# --- seed to run name -------------------------------------------------------------

def test_protocol_seed_keeps_the_config_run_name():
    """The promoted run must keep its published name, or every recorded reference to it breaks."""
    assert run_name_for("m3-fusion-lfm2.5-1.2b", protocol.SEED) == "m3-fusion-lfm2.5-1.2b"


def test_any_other_seed_suffixes_the_run_name():
    assert run_name_for("m3-fusion-lfm2.5-1.2b", 7) == "m3-fusion-lfm2.5-1.2b-seed7"


def test_seed_sweep_names_are_distinct_so_runs_cannot_clobber_each_other():
    names = {run_name_for("m3-fusion-lfm2.5-1.2b", s) for s in (protocol.SEED, 101, 202, 303)}
    assert len(names) == 4


def test_dry_run_never_resolves_to_a_real_checkpoint_directory():
    """A `--dry-run` on the promoted config must not touch the promoted checkpoint.

    This is the regression guard for the first version of the overwrite check, which
    exempted dry runs and would therefore have written a two-step adapter over
    `checkpoints/m3-fusion-lfm2.5-1.2b/final`.
    """
    for seed in (protocol.SEED, 101):
        name = run_name_for("m3-fusion-lfm2.5-1.2b", seed, dry_run=True)
        assert name.startswith("_dryrun-")
        assert name != run_name_for("m3-fusion-lfm2.5-1.2b", seed)


# --- parameter counting -----------------------------------------------------------

class _FakeParam:
    """Stands in for a torch parameter without importing torch's device machinery."""
    def __init__(self, n, requires_grad, four_bit=False):
        self._n, self.requires_grad, self.quant_state = n, requires_grad, object() if four_bit else None
        self.dtype, self._es = ("uint8" if four_bit else "bf16"), (1 if four_bit else 2)
    def numel(self): return self._n
    def element_size(self): return self._es


class _FakeModel:
    def __init__(self, params): self._p = params
    def parameters(self): return iter(self._p)


def test_four_bit_base_weights_are_counted_at_their_logical_size():
    """bitsandbytes packs two 4-bit values per byte, so `numel()` reads half the truth.

    Without the doubling a 1.2B base reports as ~600M, which would understate the ratio
    the paper's quality-per-parameter table is built on.
    """
    m = _FakeModel([_FakeParam(500_000_000, requires_grad=False, four_bit=True)])
    assert param_counts(m)["total_params"] == 1_000_000_000


def test_trainable_counts_only_adapter_weights_and_never_doubles_them():
    """LoRA weights are bf16, so the 4-bit correction must not touch them."""
    m = _FakeModel([_FakeParam(500_000_000, requires_grad=False, four_bit=True),
                    _FakeParam(33_000_000, requires_grad=True)])
    c = param_counts(m)
    assert c["trainable_params"] == 33_000_000
    assert c["total_params"] == 1_033_000_000
    assert c["trainable_pct"] == pytest.approx(3.1946, abs=1e-3)


def test_param_counts_survives_a_model_with_no_parameters():
    assert param_counts(_FakeModel([]))["trainable_pct"] is None


# --- adapter footprint ------------------------------------------------------------

def test_dir_bytes_sums_the_tree_and_sizes_the_largest_file(tmp_path):
    """Sizes are realistic on purpose: MB are rounded to 2dp, so byte-scale files round to 0.0."""
    (tmp_path / "adapter_model.safetensors").write_bytes(b"x" * 4_000_000)
    (tmp_path / "adapter_config.json").write_bytes(b"y" * 500_000)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "tokenizer.json").write_bytes(b"z" * 250_000)
    d = dir_bytes(tmp_path)
    assert d["bytes"] == 4_750_000 and d["n_files"] == 3
    assert d["mb"] == pytest.approx(4.75)
    assert d["largest_file"] == "adapter_model.safetensors"
    assert d["largest_file_mb"] == pytest.approx(4.0), \
        "the weights file alone is the figure quoted against the base model size"


def test_dir_bytes_is_none_for_an_empty_directory(tmp_path):
    """None, not 0.0. An unwritten adapter is unmeasured, not weightless."""
    assert dir_bytes(tmp_path) is None


# --- the overwrite guard ----------------------------------------------------------

def test_refuses_to_write_into_a_non_empty_checkpoint_directory(tmp_path):
    """Exercised through the CLI, because the guard must fire BEFORE the model loads.

    A guard that ran after `from_pretrained` would still cost minutes and a GPU before
    refusing, and would be easy to move below the load during a later edit.
    """
    victim = ROOT / "checkpoints" / "m3-fusion-lfm2.5-1.2b"
    if not (victim.exists() and any(victim.iterdir())):
        pytest.skip("promoted checkpoint not present in this clone")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("run_name: m3-fusion-lfm2.5-1.2b\nbase_model: nonexistent/model\n"
                   "train_file: data/built/fusion.train.jsonl\n"
                   "lora: {r: 16, alpha: 32, dropout: 0.05}\n"
                   "lr: 2.0e-4\nepochs: 3\nmicro_batch: 1\ngrad_accum: 16\nmax_seq_len: 6144\n",
                   encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "training.train_lora", "--config", str(cfg)],
                       capture_output=True, text=True, cwd=ROOT, timeout=180)
    assert p.returncode != 0
    assert "refusing to write into non-empty" in (p.stdout + p.stderr)
    assert "nonexistent/model" not in (p.stdout + p.stderr), \
        "the guard must fire before the base model is fetched"
