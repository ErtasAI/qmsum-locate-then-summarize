import json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

def setup_module():
    subprocess.run([sys.executable, "-m", "data.build_sft_m1"], check=True, cwd=ROOT)

def _rows(split):
    p = ROOT / "data" / "built" / f"sft_m1.{split}.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8")]

def test_counts_match_query_splits():
    assert len(_rows("train")) == 1257 and len(_rows("val")) == 272

def test_prompt_budget_and_schema():
    for r in _rows("train")[:50]:
        assert set(r) == {"query_id", "prompt", "response"}
        assert len(r["prompt"].split()) <= 6200  # budget + template overhead
        assert r["response"].strip()

def test_no_test_split_file_built():
    assert not (ROOT / "data" / "built" / "sft_m1.test.jsonl").exists()
