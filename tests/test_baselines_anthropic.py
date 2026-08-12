import json
import pytest
from eval import protocol
from eval.baselines_anthropic import (MODELS, build_prompt, extract_text,
                                      request_kwargs, run_split, scan_leaks)

class FakeClient:
    class messages:
        @staticmethod
        def create(**kw):
            class U: input_tokens = 1000; output_tokens = 100
            class B: type = "text"; text = "A fake summary."
            class R:
                usage = U(); content = [B()]; stop_reason = "end_turn"
                model = "claude-haiku-4-5-20251001"
            return R()

def test_build_prompt_truncates_at_explicit_budget():
    utts = [{"speaker": "A", "text": "word " * 50}] * 500
    p = build_prompt("What was decided?", utts, budget=5000)
    assert len(p.split()) < 5100 and "What was decided?" in p

@pytest.mark.parametrize("model", ["claude-opus-5", "claude-sonnet-5"])
def test_claude5_never_gets_sampling_params(model):
    """temperature/top_p/top_k return 400 on the Claude 5 models."""
    for thinking in ("off", "low"):
        kw = request_kwargs(model, thinking)
        assert not {"temperature", "top_p", "top_k"} & kw.keys()

@pytest.mark.parametrize("model", ["claude-opus-5", "claude-sonnet-5"])
def test_claude5_thinking_disabled_in_off_mode(model):
    """Thinking is on by default on Claude 5; the GPT rows are non-reasoning,
    so the default run must disable it or the comparison is not like for like."""
    assert request_kwargs(model, "off")["thinking"] == {"type": "disabled"}

def test_haiku_gets_temperature_and_no_thinking_key():
    kw = request_kwargs("claude-haiku-4-5", "off")
    assert kw["temperature"] == protocol.TEMPERATURE
    assert "thinking" not in kw

def test_thinking_low_uses_adaptive_and_more_headroom():
    kw = request_kwargs("claude-opus-5", "low")
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "low"}
    # thinking is billed against max_tokens, so the answer needs its own room
    assert kw["max_tokens"] > protocol.MAX_NEW_TOKENS

def test_off_mode_carries_anti_leak_system_prompt():
    assert "XML" in request_kwargs("claude-opus-5", "off")["system"]

def test_extract_text_joins_text_blocks_only():
    class T: type = "text"; text = "hello"
    class X: type = "thinking"; text = "should not appear"
    assert extract_text(type("R", (), {"content": [T(), X(), T()]})) == "hellohello"

def test_scan_leaks(tmp_path):
    p = tmp_path / "preds.jsonl"
    p.write_text(
        json.dumps({"query_id": "clean", "prediction": "A summary."}) + "\n" +
        json.dumps({"query_id": "leaky", "prediction": "<thinking>oops</thinking> A summary."}) + "\n",
        encoding="utf-8")
    assert scan_leaks(p) == ["leaky"]

def test_run_split_records_snapshot_and_cost(tmp_path):
    out = run_split(FakeClient(), "claude-haiku-4-5", "val", limit=2,
                    prices={"claude-haiku-4-5": {"input_per_1m": 1.0, "output_per_1m": 5.0}},
                    cache_dir=tmp_path / "cache", preds_dir=tmp_path / "preds")
    preds = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(preds) == 2 and preds[0]["prediction"] == "A fake summary."
    assert preds[0]["cost_usd"] > 0
    assert preds[0]["model_snapshot"] == "claude-haiku-4-5-20251001"
    assert preds[0]["called_at"]
    assert scan_leaks(out) == []

def test_models_list_is_current():
    assert MODELS == ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]
