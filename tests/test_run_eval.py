import json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_row_written(tmp_path):
    from data.normalize import load_queries
    queries = load_queries("val")[:3]
    pred_file = tmp_path / "preds.jsonl"
    with pred_file.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({"query_id": q["query_id"], "prediction": q["reference"],
                                "cost_usd": 0.001, "latency_s": 1.5}) + "\n")
    subprocess.run([sys.executable, "-m", "eval.run_eval", "--pred", str(pred_file),
                    "--split", "val", "--run-id", "unittest-row", "--allow-partial"],
                   check=True, cwd=ROOT)
    row = json.loads((ROOT / "results" / "rows" / "unittest-row.json").read_text())
    assert row["n"] == 3 and row["rouge1"] > 0.99
    assert row["cost_usd_total"] == 0.003
    (ROOT / "results" / "rows" / "unittest-row.json").unlink()

def test_foreign_query_ids_excluded_from_cost(tmp_path):
    from data.normalize import load_queries
    q = load_queries("val")[0]
    pred_file = tmp_path / "preds.jsonl"
    with pred_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": q["query_id"], "prediction": q["reference"],
                            "cost_usd": 0.001}) + "\n")
        f.write(json.dumps({"query_id": "not_a_real_query", "prediction": "x",
                            "cost_usd": 5.0}) + "\n")
    subprocess.run([sys.executable, "-m", "eval.run_eval", "--pred", str(pred_file),
                    "--split", "val", "--run-id", "unittest-foreign", "--allow-partial"],
                   check=True, cwd=ROOT)
    row = json.loads((ROOT / "results" / "rows" / "unittest-foreign.json").read_text())
    (ROOT / "results" / "rows" / "unittest-foreign.json").unlink()
    assert row["n"] == 1 and row["cost_usd_total"] == 0.001

def test_duplicate_query_ids_rejected(tmp_path):
    from data.normalize import load_queries
    q = load_queries("val")[0]
    pred_file = tmp_path / "preds.jsonl"
    line = json.dumps({"query_id": q["query_id"], "prediction": q["reference"]}) + "\n"
    pred_file.write_text(line + line)
    r = subprocess.run([sys.executable, "-m", "eval.run_eval", "--pred", str(pred_file),
                        "--split", "val", "--run-id", "unittest-dupe", "--allow-partial"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "duplicate query_ids" in r.stderr
    assert not (ROOT / "results" / "rows" / "unittest-dupe.json").exists()

def test_zero_matches_exits_cleanly(tmp_path):
    pred_file = tmp_path / "preds.jsonl"
    pred_file.write_text(json.dumps({"query_id": "nope", "prediction": "x"}) + "\n")
    r = subprocess.run([sys.executable, "-m", "eval.run_eval", "--pred", str(pred_file),
                        "--split", "val", "--run-id", "unittest-zero", "--allow-partial"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "nothing to score" in r.stderr
    assert not (ROOT / "results" / "rows" / "unittest-zero.json").exists()
