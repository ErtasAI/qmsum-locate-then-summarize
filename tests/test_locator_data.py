import json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

def setup_module():
    subprocess.run([sys.executable, "-m", "data.build_locator_data"], check=True, cwd=ROOT)

def _rows(split):
    return [json.loads(l) for l in (ROOT / "data" / "built" / f"locator.{split}.jsonl").open(encoding="utf-8")]

def test_labels_present_and_binary():
    rows = _rows("train")
    assert rows and all(r["label"] in (0, 1) for r in rows)
    by_query = {}
    for r in rows:
        by_query.setdefault(r["query_id"], []).append(r["label"])
    # every specific query with gold spans has at least one positive window
    assert all(any(v) for v in by_query.values())

def test_only_specific_queries_included():
    assert all("_s" in r["query_id"] for r in _rows("train"))
