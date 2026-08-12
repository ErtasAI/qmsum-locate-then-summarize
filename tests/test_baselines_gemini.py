import pytest
import json
from eval import protocol
from eval.baselines_gemini import MODELS, build_config, build_prompt, run_split

PRICES = {"gemini-3.6-flash": {"input_per_1m": 1.50, "output_per_1m": 7.50}}

class FakeClient:
    class models:
        @staticmethod
        def generate_content(model, contents, config):
            class U:
                prompt_token_count = 1000
                candidates_token_count = 100
                thoughts_token_count = 40
            class R:
                text = "A fake summary."
                usage_metadata = U()
                model_version = "gemini-3.6-flash-002"
            return R()

def test_build_prompt_truncates_at_explicit_budget():
    utts = [{"speaker": "A", "text": "word " * 50}] * 500
    p = build_prompt("What was decided?", utts, budget=5000)
    assert len(p.split()) < 5100 and "What was decided?" in p

def test_config_pins_thinking_to_the_floor_the_api_allows():
    """Gemini 3.x removed thinking_budget in favour of thinking_level and has no
    off setting: thinking_budget=0 is a 400 on gemini-3.6-flash (measured
    2026-07-27). MINIMAL is the floor, so the Gemini rows are NOT strictly
    non-reasoning like the GPT and Claude rows. Asserted here so the asymmetry
    stays visible in the code rather than becoming an unstated assumption."""
    cfg = build_config("off")
    assert cfg.thinking_config.thinking_level == "MINIMAL"
    assert cfg.thinking_config.thinking_budget is None
    assert cfg.temperature == protocol.TEMPERATURE
    assert cfg.max_output_tokens == protocol.MAX_NEW_TOKENS

def test_config_seeds_the_run():
    """The only baseline family whose API accepts a seed."""
    assert build_config("off").seed == protocol.SEED

def test_config_thinking_on_leaves_budget_unset():
    assert build_config("on").thinking_config is None

def test_run_split_records_snapshot_and_costs_thinking_tokens(tmp_path):
    out = run_split(FakeClient(), "gemini-3.6-flash", "val", limit=2, prices=PRICES,
                    cache_dir=tmp_path / "cache", preds_dir=tmp_path / "preds")
    preds = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(preds) == 2 and preds[0]["prediction"] == "A fake summary."
    assert preds[0]["model_snapshot"] == "gemini-3.6-flash-002"
    assert preds[0]["called_at"]
    # 1000 in @1.50 + (100 completion + 40 thoughts) out @7.50; thinking tokens
    # are billed as output, so omitting them would understate cost
    expected = 1000 / 1e6 * 1.50 + 140 / 1e6 * 7.50
    assert abs(preds[0]["cost_usd"] - expected) < 1e-12

def test_models_list():
    assert MODELS == ["gemini-3.6-flash", "gemini-3.5-flash"]

def test_backoff_retries_transient_errors_then_succeeds():
    """429 (free-tier cap) and 503 (model overloaded) were both observed on the
    real endpoint mid-split. Losing 281 long-context calls to either is not
    acceptable, so both must be retried rather than raised."""
    from eval.baselines_gemini import call_with_backoff
    for msg in ["429 RESOURCE_EXHAUSTED", "503 UNAVAILABLE. high demand"]:
        calls = {"n": 0}
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError(msg)
            return "ok"
        assert call_with_backoff(flaky, sleep=lambda s: None) == "ok"
        assert calls["n"] == 3

def test_backoff_does_not_swallow_a_real_request_error():
    """A 400 means the request is wrong; retrying it just wastes eight attempts
    and hides the bug."""
    from eval.baselines_gemini import call_with_backoff
    def bad():
        raise RuntimeError("400 INVALID_ARGUMENT")
    with pytest.raises(RuntimeError, match="400"):
        call_with_backoff(bad, sleep=lambda s: None)

def test_daily_quota_is_not_retried_as_if_it_were_a_rate_limit():
    """The free tier enforces a per-DAY cap and a per-minute cap, and both arrive
    as 429. Retrying the daily one burns the whole backoff chain (~10 min) per
    query and still fails, so it must be raised immediately and named."""
    from eval.baselines_gemini import DailyQuotaExhausted, call_with_backoff
    calls = {"n": 0}
    def daily():
        calls["n"] += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED ... quotaId: "
                           "GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: 20")
    with pytest.raises(DailyQuotaExhausted, match="billing"):
        call_with_backoff(daily, sleep=lambda s: None)
    assert calls["n"] == 1, "daily quota must fail on the first attempt, not retry"
