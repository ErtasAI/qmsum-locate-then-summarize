import json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_chart_includes_all_rows(tmp_path):
    row = {"run_id": "unittest-chart", "split": "val", "n": 3, "rouge1": 0.31,
           "rouge2": 0.08, "rougeL": 0.2, "rougeLsum": 0.27, "bertscore_f1": None,
           "cost_usd_total": None, "latency_s_mean": None, "notes": ""}
    p = ROOT / "results" / "rows" / "unittest-chart.json"
    p.write_text(json.dumps(row))
    try:
        subprocess.run([sys.executable, str(ROOT / "results" / "build_chart.py")], check=True)
        chart = (ROOT / "results" / "CHART.md").read_text(encoding="utf-8")
    finally:
        p.unlink()
        subprocess.run([sys.executable, str(ROOT / "results" / "build_chart.py")], check=True)
    assert "unittest-chart" in chart and "0.310" in chart
    assert "Cited numbers" in chart

def test_chart_rejects_row_missing_consumed_key():
    row = {"run_id": "unittest-badrow", "split": "val", "n": 3, "rouge1": 0.31,
           "rouge2": 0.08, "rougeLsum": 0.27}  # no bertscore_f1 / cost / latency keys
    p = ROOT / "results" / "rows" / "unittest-badrow.json"
    p.write_text(json.dumps(row))
    try:
        r = subprocess.run([sys.executable, str(ROOT / "results" / "build_chart.py")],
                           capture_output=True, text=True)
        assert r.returncode != 0
        assert "bad row file" in r.stderr
    finally:
        p.unlink()
        subprocess.run([sys.executable, str(ROOT / "results" / "build_chart.py")], check=True)
