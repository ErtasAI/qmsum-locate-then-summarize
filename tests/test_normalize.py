import json, pathlib, subprocess, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILT = ROOT / "data" / "built"

def setup_module():
    subprocess.run([sys.executable, "-m", "data.normalize"], check=True, cwd=ROOT)

def _rows(name):
    return [json.loads(l) for l in (BUILT / name).open(encoding="utf-8")]

def test_query_counts_and_schema():
    q = _rows("queries.test.jsonl")
    assert len(q) == 281  # released official split; see test_download.py note
    r = q[0]
    for key in ("query_id", "meeting_id", "split", "query_type", "query", "reference", "gold_spans"):
        assert key in r
    assert all(x["query_type"] in ("general", "specific") for x in q)

@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_gold_spans_valid(split):
    meetings = {m["meeting_id"]: m for m in _rows(f"meetings.{split}.jsonl")}
    for q in _rows(f"queries.{split}.jsonl"):
        n = len(meetings[q["meeting_id"]]["utterances"])
        for start, end in q["gold_spans"]:
            assert 0 <= start <= end < n, f"bad span {start},{end} of {n} in {q['query_id']}"

def test_no_split_leakage():
    def hashes(split):
        return {hash(" ".join(u["text"] for u in m["utterances"][:5])) for m in _rows(f"meetings.{split}.jsonl")}
    assert not hashes("train") & hashes("test")
    assert not hashes("train") & hashes("val")
    assert not hashes("val") & hashes("test")
