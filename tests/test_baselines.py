import json
import pytest
from eval import protocol
from eval.baselines_openai import (NO_TEMPERATURE, REASONING_MODELS, build_prompt,
                                   cost_usd, request_body, run_split)

class FakeClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                class D: reasoning_tokens = 0
                class U:
                    prompt_tokens = 1000; completion_tokens = 100
                    completion_tokens_details = D()
                class M: content = "A fake summary."
                class C: message = M; finish_reason = "stop"
                # `model` is the resolved snapshot the API served, not the alias
                # we requested; results rows must carry it. See pricing/ledger.
                class R: usage = U(); choices = [C()]; model = "gpt-5.4-mini-2026-01-15"
                return R()

# --- API constraints measured against the live endpoint on 2026-07-27 -----------
# These are not style preferences. Each was a 400 in a bisect probe, and encoding
# them here is what stops a future edit from silently reintroducing the failure.

@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
def test_gpt56_never_gets_temperature(model):
    """400: 'temperature does not support 0 with this model. Only the default (1)
    value is supported.' Greedy decoding is not expressible on the 5.6 tier."""
    assert model in NO_TEMPERATURE
    assert "temperature" not in request_body(model, "p")

@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"])
def test_gpt54_still_honours_the_frozen_temperature(model):
    assert request_body(model, "p")["temperature"] == protocol.TEMPERATURE

@pytest.mark.parametrize("model", sorted(REASONING_MODELS))
def test_gpt56_reasoning_is_switched_off_for_the_default_run(model):
    """The 5.6 tier reasons by default, and reasoning is charged against
    max_completion_tokens. Without this the 512-token cap is spent thinking and
    the summary comes back empty."""
    assert request_body(model, "p")["reasoning_effort"] == "none"

def test_thinking_run_is_tagged_and_gets_headroom():
    b = request_body("gpt-5.6-sol", "p", thinking="low")
    assert b["reasoning_effort"] == "low"
    assert b["max_completion_tokens"] > protocol.MAX_NEW_TOKENS

def test_build_prompt_truncates_at_explicit_budget():
    utts = [{"speaker": "A", "text": "word " * 50}] * 500
    p = build_prompt("What was decided?", utts, budget=5000)
    assert len(p.split()) < 5100 and "What was decided?" in p

def test_default_budget_leaves_qmsum_transcripts_untruncated():
    """The frontier baselines must see the FULL transcript. Longest measured
    transcript is 25,244 words (test) / 24,573 (val); dropping the budget below
    that reintroduces the handicap this protocol change removed on 2026-07-27."""
    assert protocol.PROMPT_WORD_BUDGET >= 26000

def test_cost_usd():
    prices = {"gpt-5.4-mini": {"input_per_1m": 1.0, "output_per_1m": 4.0}}
    assert abs(cost_usd("gpt-5.4-mini", 1000, 100, prices) - 0.0014) < 1e-9

def test_cost_usd_rejects_unfilled_prices():
    with pytest.raises(ValueError, match="no real prices"):
        cost_usd("gpt-5.4-mini", 1000, 100,
                 {"gpt-5.4-mini": {"input_per_1m": 0, "output_per_1m": 0}})

def test_run_split_writes_cache_and_preds(tmp_path):
    out = run_split(FakeClient(), "gpt-5.4-mini", "val", limit=2,
                    prices={"gpt-5.4-mini": {"input_per_1m": 1.0, "output_per_1m": 4.0}},
                    cache_dir=tmp_path / "cache", preds_dir=tmp_path / "preds")
    preds = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(preds) == 2 and preds[0]["prediction"] == "A fake summary."
    assert preds[0]["cost_usd"] > 0
    assert preds[0]["model_snapshot"] == "gpt-5.4-mini-2026-01-15"
    assert preds[0]["called_at"]
