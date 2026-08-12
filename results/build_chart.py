"""Assemble results/CHART.md from result rows plus cited literature numbers."""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROWS = ROOT / "results" / "rows"

CITED = """## Cited numbers (different protocols, not comparable to our rows)

| Source | Model | Number | Protocol flag |
|---|---|---|---|
| ZeroSCROLLS | GPT-4 | 18.5 | geometric mean R-1/2/L, zero-shot |
| ZeroSCROLLS | Claude (v1 era) | 14.6 | geometric mean R-1/2/L, zero-shot |
| LongBench v1 qmsum | GPT-3.5-Turbo-16k | 23.4 | ROUGE-L, subset variant |
| DialogLM paper (AAAI'22) Tab.8 | **DialogLED-large** (fine-tuned) | 34.50 R-1 | full test, own fine-tune |
| DialogLM paper (AAAI'22) Tab.3 | DialogLM base, dense | 34.02 R-1 | full test; NOT DialogLED, previously mislabelled here |
| DialogLM paper (AAAI'22) Tab.3 | DialogLM base, sparse | 33.69 R-1 | full test; the row Liu and Xu quote |
| QMSum paper | HMNet + gold spans | 36.51 R-1 | oracle locator |
"""

def main():
    rows = []
    for p in sorted(ROWS.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            for k in ("run_id", "split", "n", "rouge1", "rouge2", "rougeLsum",
                      "bertscore_f1", "cost_usd_total", "latency_s_mean"):
                r[k]
        except (json.JSONDecodeError, KeyError) as e:
            raise SystemExit(f"bad row file {p.name}: {e}")
        rows.append(r)
    rows.sort(key=lambda r: (r["split"], -r["rouge1"]))
    lines = ["# QMSum challenge chart\n",
             "| run_id | split | n | R1 | R2 | RLsum | BERTScore | cost USD | s/query | peak GB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bs = f"{r['bertscore_f1']:.3f}" if r["bertscore_f1"] is not None else ""
        cost = f"{r['cost_usd_total']:.2f}" if r["cost_usd_total"] is not None else ""
        lat = f"{r['latency_s_mean']:.1f}" if r["latency_s_mean"] is not None else ""
        # Read with .get(): rows written before eval/vram.py existed have no such key, and
        # promoting it to a required key would make every historical row fail validation.
        # Blank means "not instrumented", never zero.
        vram_max = r.get("peak_vram_gb_max")
        gb = f"{vram_max:.2f}" if vram_max is not None else ""
        lines.append(f"| {r['run_id']} | {r['split']} | {r['n']} | {r['rouge1']:.3f} | "
                     f"{r['rouge2']:.3f} | {r['rougeLsum']:.3f} | {bs} | {cost} | {lat} | {gb} |")
    (ROOT / "results" / "CHART.md").write_text("\n".join(lines) + "\n\n" + CITED, encoding="utf-8")
    print(f"chart: {len(rows)} rows")

if __name__ == "__main__":
    main()
