import json
import random
import pytest
from data.normalize import load_queries
from eval import protocol
from eval.judge import (JUDGE_MAX_TOKENS, build_judge_prompt, cache_key,
                        call_with_retry, is_transient, parse_verdict,
                        pick_sample, request_kwargs_claude, request_kwargs_gpt,
                        run_judgement, vendor_of, vendor_of_preds)

# Anything that could identify the system behind a summary.
IDENTIFIERS = ["gpt", "openai", "claude", "anthropic", "opus", "sonnet", "haiku",
               "gemini", "google", "flash", "lfm", "liquid", "qwen", "distilbart",
               "pegasus", "bart", "dialogled", "ertas", "baseline", "ours", "m3-fusion"]

def assert_blind(prompt):
    low = prompt.lower()
    leaked = [t for t in IDENTIFIERS if t in low]
    assert not leaked, f"judge prompt leaked identifiers: {leaked}"

def _preds_file(tmp_path, name, qids, text, snapshot):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(
        {"query_id": q, "prediction": text, "model_snapshot": snapshot}) for q in qids),
        encoding="utf-8")
    return p

class RecordingClient:
    """Captures every prompt the judge sends so blinding can be asserted end to end."""
    def __init__(self, verdict="A"):
        self.prompts = []; self.kwargs = []; self.verdict = verdict
    class _Completions:
        def __init__(self, outer): self.outer = outer
        def create(self, model, messages, **kw):
            self.outer.prompts.append(messages[0]["content"])
            self.outer.kwargs.append(kw)
            class U: prompt_tokens = 500; completion_tokens = 120
            class M: content = f"Reasoning here.\nVERDICT: {self.outer.verdict}"
            class C: message = M(); finish_reason = "stop"
            class R: choices = [C()]; model = "gpt-5.6-sol-2026-07-09"; usage = U()
            return R()
    @property
    def chat(self):
        return type("chat", (), {"completions": RecordingClient._Completions(self)})()

class ClaudeRecordingClient:
    """Anthropic-shaped mock: typed content blocks, input/output token usage."""
    def __init__(self, verdict="A", stop_reason="end_turn"):
        self.prompts = []; self.kwargs = []
        self.verdict = verdict; self.stop_reason = stop_reason
    class _Messages:
        def __init__(self, outer): self.outer = outer
        def create(self, model, messages, **kw):
            self.outer.prompts.append(messages[0]["content"])
            self.outer.kwargs.append(kw)
            o = self.outer
            class B: type = "text"; text = f"Reasoning here.\nVERDICT: {o.verdict}"
            class U: input_tokens = 500; output_tokens = 120
            class R:
                content = [B()]; model = "claude-opus-5-20260701"
                usage = U(); stop_reason = o.stop_reason
            return R()
    @property
    def messages(self):
        return ClaudeRecordingClient._Messages(self)

# --- existing coverage ------------------------------------------------------

def test_prompt_is_blind_and_contains_both():
    p = build_judge_prompt("What was decided?", "Summary alpha.", "Summary beta.")
    assert "Summary alpha." in p and "Summary beta." in p
    assert "gpt" not in p.lower() and "ertas" not in p.lower()

def test_parse_verdict():
    assert parse_verdict("VERDICT: A") == "A"
    assert parse_verdict("...reasoning...\nVERDICT: TIE") == "TIE"

def test_parse_verdict_unparsed():
    assert parse_verdict("no verdict line here") == "UNPARSED"

def test_pick_sample_deterministic():
    ids = [f"q{i}" for i in range(300)]
    assert pick_sample(ids, 60) == pick_sample(ids, 60)
    assert len(pick_sample(ids, 60)) == 60

# --- blinding ---------------------------------------------------------------

def test_prompt_builder_is_blind_against_full_identifier_list():
    assert_blind(build_judge_prompt("What was decided?", "Summary one.", "Summary two."))

def test_prompt_uses_anonymous_labels():
    p = build_judge_prompt("q", "AAA", "BBB")
    assert "Summary A:" in p and "Summary B:" in p and "anonymous" in p

def test_every_dispatched_prompt_is_blind(tmp_path):
    """End to end: no identifier reaches the judge, even though the predictions
    files themselves record which model produced them."""
    qids = [q["query_id"] for q in load_queries("val")][:8]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha summary.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta summary.", "gpt-5.6-sol-2026-07-09")
    client = RecordingClient()
    run_judgement(a, b, "val", 6, client, provider="gpt")
    assert len(client.prompts) == 6
    for prompt in client.prompts:
        assert_blind(prompt)

def test_claude_judge_prompts_are_blind_too(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:8]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha summary.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta summary.", "gpt-5.6-sol-2026-07-09")
    client = ClaudeRecordingClient()
    run_judgement(a, b, "val", 6, client, provider="claude")
    assert len(client.prompts) == 6
    for prompt in client.prompts:
        assert_blind(prompt)

# --- grounded mode ----------------------------------------------------------

def test_grounded_prompt_shows_the_reference_and_stays_blind():
    p = build_judge_prompt("What was decided?", "Alpha.", "Beta.",
                           reference="The group agreed to defer the vote.")
    assert "The group agreed to defer the vote." in p
    assert "Reference answer:" in p
    assert_blind(p)

def test_grounded_prompt_denies_credit_for_material_outside_the_reference():
    """The verbosity bias is in the prompt's control, not just the analysis: a
    grounded judge that still rewarded extra detail would measure length again."""
    p = build_judge_prompt("q", "A", "B", reference="R")
    assert "not a merit" in p

def test_preference_prompt_shows_no_reference():
    p = build_judge_prompt("What was decided?", "Alpha.", "Beta.")
    assert "Reference answer:" not in p

def test_grounded_mode_reaches_the_api_with_the_reference(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:4]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta.", "claude-opus-5-20260701")
    client = RecordingClient()
    t = run_judgement(a, b, "val", 4, client, provider="gpt", mode="grounded")
    assert t["mode"] == "grounded"
    assert all("Reference answer:" in p for p in client.prompts)

def test_preference_mode_is_the_default(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:4]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta.", "claude-opus-5-20260701")
    client = RecordingClient()
    t = run_judgement(a, b, "val", 4, client, provider="gpt")
    assert t["mode"] == "preference"
    assert all("Reference answer:" not in p for p in client.prompts)

def test_unknown_mode_fails_loudly(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:2]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta.", "claude-opus-5-20260701")
    with pytest.raises(ValueError):
        run_judgement(a, b, "val", 2, RecordingClient(), provider="gpt", mode="freeform")

# --- position swap ----------------------------------------------------------

def test_position_swap_cancels_a_constant_verdict(tmp_path):
    """A judge that always answers 'A' must split evenly once positions are
    swapped on alternating items, or position bias gets scored as skill."""
    qids = [q["query_id"] for q in load_queries("val")][:20]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha summary.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta summary.", "gpt-5.6-sol-2026-07-09")
    t = run_judgement(a, b, "val", 20, RecordingClient("A"), provider="gpt")
    assert t["a_wins"] == 10 and t["b_wins"] == 10

# --- protocol knobs the 2026 frontier tier rejects --------------------------
# Locked against the measured API behaviour (2026-07-27). Sending temperature to
# a model that 400s on it would fail the whole panel on the first call.

@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra"])
def test_gpt_judge_sends_no_temperature_and_disables_reasoning(model):
    kw = request_kwargs_gpt(model)
    assert "temperature" not in kw
    assert kw["reasoning_effort"] == "none"

def test_gpt_judge_keeps_temperature_where_the_api_accepts_it():
    assert request_kwargs_gpt("gpt-5.4")["temperature"] == protocol.TEMPERATURE

@pytest.mark.parametrize("model", ["claude-opus-5", "claude-sonnet-5"])
def test_claude_judge_sends_no_temperature_and_disables_thinking(model):
    kw = request_kwargs_claude(model)
    assert "temperature" not in kw
    assert kw["thinking"] == {"type": "disabled"}
    assert kw["max_tokens"] == JUDGE_MAX_TOKENS

def test_claude_haiku_judge_keeps_temperature_and_sends_no_thinking():
    kw = request_kwargs_claude("claude-haiku-4-5")
    assert kw["temperature"] == protocol.TEMPERATURE
    assert "thinking" not in kw

def test_dispatched_gpt_call_carries_the_protocol_knobs(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:4]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta.", "claude-opus-5-20260701")
    client = RecordingClient()
    run_judgement(a, b, "val", 4, client, provider="gpt")
    for kw in client.kwargs:
        assert "temperature" not in kw
        assert kw["reasoning_effort"] == "none"
        assert kw["max_completion_tokens"] == JUDGE_MAX_TOKENS

def test_judge_budget_is_not_the_scored_generation_budget():
    """A judge writes reasoning then a verdict; capping it at the 512-token
    answer budget would truncate the VERDICT line into an UNPARSED row."""
    assert JUDGE_MAX_TOKENS > protocol.MAX_NEW_TOKENS

# --- unparsed is not a tie ---------------------------------------------------

def test_unparsed_is_not_counted_as_a_tie(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:4]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta.", "claude-opus-5-20260701")
    class NoVerdict(RecordingClient):
        class _C:
            def __init__(self, outer): self.outer = outer
            def create(self, model, messages, **kw):
                self.outer.prompts.append(messages[0]["content"])
                class U: prompt_tokens = 10; completion_tokens = 10
                class M: content = "I have no opinion."
                class C: message = M(); finish_reason = "length"
                class R: choices = [C()]; model = "gpt-5.6-sol-2026-07-09"; usage = U()
                return R()
        @property
        def chat(self):
            return type("chat", (), {"completions": NoVerdict._C(self)})()
    t = run_judgement(a, b, "val", 4, NoVerdict(), provider="gpt")
    assert t["unparsed"] == 4
    assert t["ties"] == 0
    assert t["n_decided"] == 0
    assert t["truncated"] == 4
    assert all(i["winner"] is None for i in t["items"])

# --- transient-failure retry -------------------------------------------------
# An unretried 529 killed a full preference-panel run on 2026-07-27 after one
# completed leg. Hundreds of sequential synchronous calls make this certain, not
# unlucky.

class _Boom(Exception):
    def __init__(self, status=None): self.status_code = status

class OverloadedError(Exception): pass
class RateLimitError(Exception): pass
class APIConnectionError(Exception): pass

def test_transient_detected_by_status_code():
    assert all(is_transient(_Boom(s)) for s in (429, 500, 502, 503, 529))

def test_transient_detected_by_exception_name():
    """Vendor SDKs raise typed errors without a status attribute; the class name
    is the only signal available without importing both SDKs."""
    assert is_transient(OverloadedError())
    assert is_transient(RateLimitError())
    assert is_transient(APIConnectionError())

def test_client_errors_are_not_transient():
    """A 400 is a bug in the request body. Retrying it six times hides it."""
    assert not is_transient(_Boom(400))
    assert not is_transient(_Boom(401))
    assert not is_transient(ValueError("bad"))

def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3: raise OverloadedError()
        return "ok"
    slept = []
    assert call_with_retry(flaky, sleep=slept.append, rng=random.Random(0)) == "ok"
    assert calls["n"] == 3 and len(slept) == 2

def test_backoff_grows_between_attempts():
    def always(): raise OverloadedError()
    slept = []
    with pytest.raises(OverloadedError):
        call_with_retry(always, attempts=4, sleep=slept.append, rng=random.Random(0))
    assert len(slept) == 3
    assert slept[-1] > slept[0]

def test_non_transient_error_is_raised_without_retrying():
    calls = {"n": 0}
    def bad():
        calls["n"] += 1
        raise ValueError("bad request")
    with pytest.raises(ValueError):
        call_with_retry(bad, sleep=lambda s: None)
    assert calls["n"] == 1

def test_retry_gives_up_and_reraises_the_real_error():
    def always(): raise OverloadedError("still overloaded")
    with pytest.raises(OverloadedError):
        call_with_retry(always, attempts=2, sleep=lambda s: None)

def test_judgement_survives_a_transient_failure(tmp_path):
    """End to end: a flaky client must not lose the run."""
    qids = [q["query_id"] for q in load_queries("val")][:4]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta.", "claude-opus-5-20260701")
    class Flaky(RecordingClient):
        fails = {"n": 0}
        class _C:
            def __init__(self, outer): self.outer = outer
            def create(self, model, messages, **kw):
                Flaky.fails["n"] += 1
                if Flaky.fails["n"] == 1: raise OverloadedError()
                class U: prompt_tokens = 10; completion_tokens = 10
                class M: content = "VERDICT: A"
                class C: message = M(); finish_reason = "stop"
                class R: choices = [C()]; model = "gpt-5.6-sol-2026-07-09"; usage = U()
                return R()
        @property
        def chat(self):
            return type("chat", (), {"completions": Flaky._C(self)})()
    import eval.judge as _J
    orig, _J.time.sleep = _J.time.sleep, lambda s: None
    try:
        t = run_judgement(a, b, "val", 4, Flaky(), provider="gpt")
    finally:
        _J.time.sleep = orig
    assert t["n_decided"] == 4 and t["unparsed"] == 0

# --- caching -----------------------------------------------------------------

def test_cache_key_encodes_orientation():
    assert cache_key("q1", False) != cache_key("q1", True)

def test_cached_judgements_are_never_rebilled(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:6]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta.", "claude-opus-5-20260701")
    cache = tmp_path / "cache"
    first = RecordingClient()
    t1 = run_judgement(a, b, "val", 6, first, provider="gpt", cache_dir=cache)
    second = RecordingClient()
    t2 = run_judgement(a, b, "val", 6, second, provider="gpt", cache_dir=cache)
    assert len(first.prompts) == 6
    assert second.prompts == [], "a cached judgement was re-sent to the API"
    assert (t1["a_wins"], t1["b_wins"]) == (t2["a_wins"], t2["b_wins"])

def test_cost_is_recorded_from_usage(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:4]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta.", "claude-opus-5-20260701")
    t = run_judgement(a, b, "val", 4, RecordingClient(), provider="gpt")
    assert t["cost_usd"] > 0
    assert t["entries_without_usage"] == 0

def test_missing_usage_is_absent_not_zero(tmp_path):
    """A response with no usage must not silently cost $0.00 in the tally."""
    class NoUsage(RecordingClient):
        class _C:
            def __init__(self, outer): self.outer = outer
            def create(self, model, messages, **kw):
                class M: content = "VERDICT: A"
                class C: message = M(); finish_reason = "stop"
                class R: choices = [C()]; model = "gpt-5.6-sol-2026-07-09"
                return R()
        @property
        def chat(self):
            return type("chat", (), {"completions": NoUsage._C(self)})()
    qids = [q["query_id"] for q in load_queries("val")][:4]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta.", "claude-opus-5-20260701")
    t = run_judgement(a, b, "val", 4, NoUsage(), provider="gpt")
    assert t["entries_without_usage"] == 4
    assert t["cost_usd"] == 0.0

# --- self-preference detection ---------------------------------------------

@pytest.mark.parametrize("model,vendor", [
    ("gpt-5.6-sol", "openai"), ("claude-opus-5", "anthropic"),
    ("gemini-3.6-flash", "google"), ("lfm2.5-1.2b", None)])
def test_vendor_of(model, vendor):
    assert vendor_of(model) == vendor

def test_vendor_of_preds_reads_snapshot(tmp_path):
    f = _preds_file(tmp_path, "p.jsonl", ["q1"], "x", "claude-opus-5-20260701")
    assert vendor_of_preds(f) == "anthropic"

def test_gpt_judge_flags_self_preference_against_a_gpt_contestant(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:4]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha summary.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta summary.", "gpt-5.6-sol-2026-07-09")
    t = run_judgement(a, b, "val", 4, RecordingClient(), provider="gpt")
    assert t["self_preference_risk"] is True
    assert t["judge_vendor"] == "openai" and "openai" in t["contestant_vendors"]

def test_gpt_judge_is_clean_against_a_non_gpt_contestant(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:4]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha summary.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta summary.", "claude-opus-5-20260701")
    t = run_judgement(a, b, "val", 4, RecordingClient(), provider="gpt")
    assert t["self_preference_risk"] is False

def test_claude_judge_flags_self_preference_against_a_claude_contestant(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:4]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta.", "claude-opus-5-20260701")
    t = run_judgement(a, b, "val", 4, ClaudeRecordingClient(), provider="claude")
    assert t["self_preference_risk"] is True
    assert t["judge_vendor"] == "anthropic"

def test_claude_judge_is_clean_against_a_gpt_contestant(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:4]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta.", "gpt-5.6-luna-2026-07-09")
    t = run_judgement(a, b, "val", 4, ClaudeRecordingClient(), provider="claude")
    assert t["self_preference_risk"] is False

def test_our_own_system_never_triggers_a_vendor_clash(tmp_path):
    """The winner is an LFM2.5 fine-tune on QMSum gold: neither judge's vendor.
    If this ever fails, a judge is scoring output from its own house."""
    qids = [q["query_id"] for q in load_queries("val")][:2]
    ours = _preds_file(tmp_path, "ours.jsonl", qids, "Alpha.", "lfm2.5-1.2b-local")
    assert vendor_of_preds(ours) is None

def test_judge_metadata_recorded(tmp_path):
    qids = [q["query_id"] for q in load_queries("val")][:4]
    a = _preds_file(tmp_path, "a.jsonl", qids, "Alpha summary.", "lfm2.5-1.2b-local")
    b = _preds_file(tmp_path, "b.jsonl", qids, "Beta summary.", "claude-opus-5-20260701")
    t = run_judgement(a, b, "val", 4, RecordingClient(), provider="gpt")
    assert t["judge_model"] == "gpt-5.6-sol"
    assert t["judge_snapshot"] == "gpt-5.6-sol-2026-07-09"
    assert t["judged_at"]
