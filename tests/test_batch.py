"""Batch-API path: request shape, result parsing, and the pricing consequence.

The load-bearing property here is that a cache holding BOTH synchronous and
batched responses prices each row at what was actually paid for it, rather than
at whatever mode the current run happens to be in.
"""
import json
import pathlib
import pytest
from data.normalize import load_queries
from eval import batch as B
from eval import baselines_anthropic as A
from eval import baselines_openai as O
from eval.pricing import cost_usd

PRICES = {
    "gpt-5.6-sol": {"input_per_1m": 5.0, "output_per_1m": 30.0,
                    "input_per_1m_batch": 2.5, "output_per_1m_batch": 15.0},
    "claude-opus-5": {"input_per_1m": 5.0, "output_per_1m": 25.0,
                      "input_per_1m_batch": 2.5, "output_per_1m_batch": 12.5},
}

# --- pricing ----------------------------------------------------------------

def test_batch_rate_is_half_the_sync_rate():
    sync = cost_usd("gpt-5.6-sol", 1_000_000, 100_000, PRICES)
    bat = cost_usd("gpt-5.6-sol", 1_000_000, 100_000, PRICES, batch=True)
    assert abs(bat - sync / 2) < 1e-12

def test_missing_batch_rates_fail_loudly_rather_than_guessing_a_halving():
    with pytest.raises(ValueError, match="no batch rates"):
        cost_usd("m", 100, 10, {"m": {"input_per_1m": 1.0, "output_per_1m": 2.0}}, batch=True)

def test_preds_row_prices_from_the_entry_not_the_run():
    """A cache mixing a synchronous smoke test with a batched full split must not
    bill the synchronous rows at the discounted rate."""
    e = {"prediction": "x", "prompt_tokens": 1_000_000, "completion_tokens": 0}
    sync = B.preds_row("gpt-5.6-sol", "q1", {**e, "mode": "sync"}, PRICES)
    bat = B.preds_row("gpt-5.6-sol", "q1", {**e, "mode": "batch"}, PRICES)
    assert sync["cost_usd"] == 5.0 and bat["cost_usd"] == 2.5
    assert sync["mode"] == "sync" and bat["mode"] == "batch"

def test_preds_row_defaults_to_sync_for_pre_batch_cache_entries():
    """Entries written before mode existed must never be assumed discounted."""
    row = B.preds_row("gpt-5.6-sol", "q1",
                      {"prediction": "x", "prompt_tokens": 1_000_000, "completion_tokens": 0}, PRICES)
    assert row["mode"] == "sync" and row["cost_usd"] == 5.0

def test_preds_row_bills_gemini_thinking_tokens_as_output():
    p = {"g": {"input_per_1m": 1.0, "output_per_1m": 10.0}}
    row = B.preds_row("g", "q1", {"prediction": "x", "prompt_tokens": 2000,
                                  "completion_tokens": 100, "thoughts_tokens": 900}, p)
    # 900 thinking tokens are billed as output; dropping them understates by 90%
    assert abs(row["cost_usd"] - (2000 / 1e6 * 1.0 + 1000 / 1e6 * 10.0)) < 1e-12

# --- job bookkeeping --------------------------------------------------------

def test_run_tag_marks_deviations_so_they_cannot_overwrite_the_protocol_run():
    from eval import protocol
    assert B.run_tag("claude-opus-5", protocol.PROMPT_WORD_BUDGET) == "claude-opus-5"
    assert B.run_tag("claude-opus-5", 14000) == "claude-opus-5-w14000"
    assert B.run_tag("claude-opus-5", None, "low") == "claude-opus-5-thinklow"

def test_pending_excludes_already_cached_queries(tmp_path):
    ids = [q["query_id"] for q in load_queries("val")[:5]]
    (tmp_path / f"{ids[0]}.json").write_text("{}", encoding="utf-8")
    assert [q["query_id"] for q in B.pending("val", tmp_path, limit=5)] == ids[1:]

def test_write_preds_reports_missing_rather_than_faking_them(tmp_path):
    ids = [q["query_id"] for q in load_queries("val")[:4]]
    for i in ids[:2]:
        (tmp_path / f"{i}.json").write_text(json.dumps(
            {"prediction": "s", "prompt_tokens": 10, "completion_tokens": 1, "mode": "batch"}),
            encoding="utf-8")
    out, written, missing = B.write_preds("gpt-5.6-sol", "val", "t", PRICES, limit=4,
                                          cache_dir=tmp_path, preds_dir=tmp_path / "p")
    assert (written, missing) == (2, 2)
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2

# --- OpenAI batch -----------------------------------------------------------

class FakeOpenAI:
    def __init__(self, output_lines=()):
        self.submitted = None
        self._out = "\n".join(json.dumps(l) for l in output_lines)
        outer = self
        class Files:
            def create(self, file, purpose):
                outer.submitted = file[1]
                return type("F", (), {"id": "file_1"})()
            def content(self, fid):
                return type("C", (), {"text": outer._out})()
        class Batches:
            def create(self, **kw):
                outer.batch_kw = kw
                return type("Bt", (), {"id": "batch_1", "status": "validating"})()
            def retrieve(self, bid):
                return type("Bt", (), {
                    "status": "completed", "output_file_id": "out_1", "error_file_id": None,
                    "request_counts": type("RC", (), {"completed": 2, "failed": 0, "total": 2})()})()
        self.files, self.batches = Files(), Batches()

def _openai_out(cid, content="A batched summary."):
    return {"custom_id": cid, "error": None,
            "response": {"status_code": 200, "body": {
                "model": "gpt-5.6-sol-2026-07-09",
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}}}

def test_openai_submit_builds_one_jsonl_line_per_pending_query(tmp_path):
    job = O.submit_batch(FakeOpenAI(), "gpt-5.6-sol", "val", limit=3,
                         cache_dir=tmp_path / "c", jobs_dir=tmp_path / "j")
    assert job["n_requests"] == 3 and job["provider"] == "openai"
    assert (tmp_path / "j" / f"{job['tag']}-val.json").exists()

def test_openai_batch_body_matches_the_synchronous_request():
    """Batch and sync must send the same scored request or the rows are not
    comparable and the discount is measuring a different experiment."""
    body = O.request_body("gpt-5.6-sol", "PROMPT")
    assert body["max_completion_tokens"] == 512
    assert body["messages"] == [{"role": "user", "content": "PROMPT"}]

def test_openai_submit_is_a_noop_when_the_cache_is_complete(tmp_path):
    c = tmp_path / "c"; c.mkdir()
    for q in load_queries("val")[:2]:
        (c / f"{q['query_id']}.json").write_text("{}", encoding="utf-8")
    assert O.submit_batch(FakeOpenAI(), "gpt-5.6-sol", "val", limit=2,
                          cache_dir=c, jobs_dir=tmp_path / "j") is None

def test_openai_collect_caches_successes_and_tags_them_batch(tmp_path):
    ids = [q["query_id"] for q in load_queries("val")[:2]]
    cl = FakeOpenAI([_openai_out(i) for i in ids])
    job = {"provider": "openai", "batch_id": "b", "model": "gpt-5.6-sol", "split": "val",
           "tag": "gpt-5.6-sol", "budget": 40000, "n_requests": 2, "cache_dir": str(tmp_path)}
    state, written, errors = O.collect_batch(cl, job)
    assert (state["status"], written, errors) == ("completed", 2, [])
    c = json.loads((tmp_path / f"{ids[0]}.json").read_text(encoding="utf-8"))
    assert c["mode"] == "batch" and c["model_snapshot"] == "gpt-5.6-sol-2026-07-09"
    assert c["latency_s"] is None      # batch has no meaningful per-query latency

def test_openai_collect_does_not_cache_failures_so_a_resubmit_retries_them(tmp_path):
    ids = [q["query_id"] for q in load_queries("val")[:2]]
    bad = {"custom_id": ids[1], "error": {"message": "boom"}, "response": None}
    state, written, errors = O.collect_batch(
        FakeOpenAI([_openai_out(ids[0]), bad]),
        {"provider": "openai", "batch_id": "b", "model": "gpt-5.6-sol", "split": "val",
         "tag": "gpt-5.6-sol", "budget": 40000, "n_requests": 2, "cache_dir": str(tmp_path)})
    assert written == 1 and len(errors) == 1
    assert not (tmp_path / f"{ids[1]}.json").exists()
    assert [q["query_id"] for q in B.pending("val", tmp_path, limit=2)] == [ids[1]]

# --- Anthropic batch --------------------------------------------------------

class FakeMsg:
    def __init__(self, text="A batched summary.", stop="end_turn"):
        self.content = [type("T", (), {"type": "text", "text": text})()]
        self.usage = type("U", (), {"input_tokens": 1000, "output_tokens": 100})()
        self.model, self.stop_reason = "claude-opus-5-20260701", stop

class FakeAnthropic:
    def __init__(self, results=()):
        outer = self
        class Batches:
            def create(self, requests):
                outer.requests = requests
                return type("B", (), {"id": "msgbatch_1", "processing_status": "in_progress"})()
            def retrieve(self, bid):
                return type("B", (), {"processing_status": "ended",
                    "request_counts": type("RC", (), {"succeeded": 2, "errored": 0, "expired": 0})()})()
            def results(self, bid):
                return iter(results)
        self.messages = type("M", (), {"batches": Batches()})()

def _entry(cid, msg=None, kind="succeeded"):
    res = type("R", (), {"type": kind, "message": msg})()
    return type("E", (), {"custom_id": cid, "result": res})()

def test_anthropic_submit_sends_the_same_params_as_the_sync_path(tmp_path):
    cl = FakeAnthropic()
    O_kw = A.request_kwargs("claude-opus-5", "off")
    A.submit_batch(cl, "claude-opus-5", "val", limit=2,
                   cache_dir=tmp_path / "c", jobs_dir=tmp_path / "j")
    p = cl.requests[0]["params"]
    assert p["model"] == "claude-opus-5"
    assert p["thinking"] == O_kw["thinking"] == {"type": "disabled"}
    assert "temperature" not in p          # 400 on Claude 5
    assert p["system"] == A.ANTI_LEAK_SYSTEM
    assert cl.requests[0]["custom_id"] == load_queries("val")[0]["query_id"]

def test_anthropic_collect_caches_successes(tmp_path):
    ids = [q["query_id"] for q in load_queries("val")[:2]]
    cl = FakeAnthropic([_entry(i, FakeMsg()) for i in ids])
    job = {"provider": "anthropic", "batch_id": "b", "model": "claude-opus-5", "split": "val",
           "tag": "claude-opus-5", "budget": 40000, "thinking": "off", "n_requests": 2,
           "cache_dir": str(tmp_path)}
    state, written, errors = A.collect_batch(cl, job)
    assert (state["status"], written, errors) == ("ended", 2, [])
    c = json.loads((tmp_path / f"{ids[0]}.json").read_text(encoding="utf-8"))
    assert c["mode"] == "batch" and c["latency_s"] is None
    assert c["model_snapshot"] == "claude-opus-5-20260701"

@pytest.mark.parametrize("entry_kind,stop", [("errored", "end_turn"), ("expired", "end_turn"),
                                             ("succeeded", "refusal")])
def test_anthropic_collect_never_caches_a_non_answer(tmp_path, entry_kind, stop):
    """An errored, expired or refused request must stay uncached, or a resubmit
    would skip it and the split would be silently short."""
    qid = load_queries("val")[0]["query_id"]
    cl = FakeAnthropic([_entry(qid, FakeMsg(stop=stop), entry_kind)])
    _, written, errors = A.collect_batch(
        cl, {"provider": "anthropic", "batch_id": "b", "model": "claude-opus-5", "split": "val",
             "tag": "claude-opus-5", "budget": 40000, "thinking": "off", "n_requests": 1,
             "cache_dir": str(tmp_path)})
    assert written == 0 and len(errors) == 1
    assert not (tmp_path / f"{qid}.json").exists()

# --- enqueued-token chunking ------------------------------------------------

def test_openai_chunks_to_stay_under_the_enqueued_token_ceiling(tmp_path):
    """OpenAI rejected the whole 281-request split with 'Enqueued token limit
    reached ... Limit: 900,000'. A chunk must fit under the cap and report what
    is left, or the batch fails outright and nothing runs."""
    job = O.submit_batch(FakeOpenAI(), "gpt-5.6-sol", "test",
                         cache_dir=tmp_path / "c", jobs_dir=tmp_path / "j")
    cap = O.ENQUEUED_TOKEN_LIMIT["gpt-5.6-sol"] * O.ENQUEUE_SAFETY
    assert job["est_enqueued_tokens"] <= cap
    assert job["n_requests"] < 281 and job["remaining"] > 0
    assert job["n_requests"] + job["remaining"] == 281

def test_luna_gets_a_bigger_chunk_than_sol_because_its_ceiling_is_higher(tmp_path):
    sol = O.submit_batch(FakeOpenAI(), "gpt-5.6-sol", "test",
                         cache_dir=tmp_path / "cs", jobs_dir=tmp_path / "j")
    luna = O.submit_batch(FakeOpenAI(), "gpt-5.6-luna", "test",
                          cache_dir=tmp_path / "cl", jobs_dir=tmp_path / "j")
    assert luna["n_requests"] > sol["n_requests"]

def test_token_estimate_is_conservative_against_the_measured_ratio():
    """Measured 4.75 chars/token on QMSum prompts. The estimator must not
    UNDER-count, because crossing the ceiling fails the entire batch."""
    assert O.CHARS_PER_TOKEN < 4.75
    assert O.estimate_tokens("x" * 47500) >= 10000

def _submitted_ids(client):
    """custom_ids in the JSONL the client actually uploaded."""
    return [json.loads(l)["custom_id"]
            for l in client.submitted.decode("utf-8").splitlines() if l.strip()]

def test_second_chunk_covers_exactly_what_the_first_did_not(tmp_path):
    """After collecting chunk 1, the next submit must pick up the remainder and
    never re-bill a query already in the cache. Chunk boundaries must therefore
    partition the split, with no overlap and no gap."""
    c, j = tmp_path / "c", tmp_path / "j"
    c1 = FakeOpenAI(); O.submit_batch(c1, "gpt-5.6-sol", "test", cache_dir=c, jobs_dir=j)
    first = _submitted_ids(c1)
    for cid in first:                                  # simulate collecting chunk 1
        (c / f"{cid}.json").write_text("{}", encoding="utf-8")

    c2 = FakeOpenAI(); O.submit_batch(c2, "gpt-5.6-sol", "test", cache_dir=c, jobs_dir=j)
    second = _submitted_ids(c2)

    assert second, "second chunk must not be empty"
    assert not set(first) & set(second), "chunks must not overlap (would re-bill)"
    all_ids = [q["query_id"] for q in load_queries("test")]
    assert first + second == all_ids[:len(first) + len(second)], "chunks must not skip a query"

def test_every_request_in_a_chunk_carries_the_scored_body(tmp_path):
    cl = FakeOpenAI()
    O.submit_batch(cl, "gpt-5.6-sol", "test", limit=3, cache_dir=tmp_path / "c",
                   jobs_dir=tmp_path / "j")
    for line in cl.submitted.decode("utf-8").splitlines():
        rec = json.loads(line)
        assert rec["url"] == O.BATCH_ENDPOINT and rec["method"] == "POST"
        assert rec["body"]["reasoning_effort"] == "none"
        assert "temperature" not in rec["body"]
        assert "Transcript:" in rec["body"]["messages"][0]["content"]

# --- latency accounting -----------------------------------------------------

def test_warmup_query_is_excluded_from_the_latency_mean(tmp_path):
    """The first query carries CUDA context creation and kernel autotuning and runs
    several times slower than steady state. Averaging it in overstates latency."""
    import subprocess, sys, os
    rows = [{"query_id": q["query_id"], "prediction": "s",
             "latency_s": 30.0 if i == 0 else 3.0,
             "locate_s": 0.5 if i == 0 else 0.02,
             "generate_s": 29.5 if i == 0 else 2.98,
             "warmup": i == 0, "device": "TestGPU"}
            for i, q in enumerate(load_queries("val")[:4])]
    p = tmp_path / "preds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(__file__).resolve().parents[1])}
    subprocess.run([sys.executable, "-m", "eval.run_eval", "--pred", str(p),
                    "--split", "val", "--run-id", "tmp-warmup-test", "--allow-partial"],
                   cwd=env["PYTHONPATH"], env=env, check=True, capture_output=True)
    row = json.loads((pathlib.Path(env["PYTHONPATH"]) / "results" / "rows"
                      / "tmp-warmup-test.json").read_text())
    (pathlib.Path(env["PYTHONPATH"]) / "results" / "rows" / "tmp-warmup-test.json").unlink()
    assert row["latency_s_mean"] == 3.0, "warmup row leaked into the mean"
    assert row["latency_n_timed"] == 3
    assert row["locate_s_mean"] == 0.02 and row["generate_s_mean"] == 2.98
    assert row["device"] == "TestGPU"
