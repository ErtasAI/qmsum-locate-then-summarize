"""Gap 1 backfill: populate `cost_usd_total` in results/rows/*.json.

Sources, in order of authority:
- The five frontier rows carry the batch costs recorded at collect time
  (2026-07-27): luna 2.05, sol 10.23, opus 16.20, haiku 2.35, sonnet 6.50,
  total 37.33.
- Rows we generated on the local GPU get 0.00 (no API in the loop).
- The two `row-socratic-segenc-*` rows stay null: no inference was run for
  them at all, ours or anyone's, so a marginal cost is not defined.

Dry-run by default; pass --apply to write. Latency stays untouched:
batched API rows carry no per-query latency by construction, and local
rows already record theirs.
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROWS = ROOT / "results" / "rows"

FRONTIER_COSTS = {
    "baseline-gpt-5.6-luna-test": 2.05,
    "baseline-gpt-5.6-sol-test": 10.23,
    "baseline-claude-opus-5-test": 16.20,
    "baseline-claude-haiku-4-5-test": 2.35,
    "baseline-claude-sonnet-5-test": 6.50,
}
FRONTIER_NOTE = ("cost_usd_total backfilled 2026-08-04 from the batch cost recorded at "
                 "collect time (batch cost records, 2026-07-27)")
LOCAL_NOTE = "cost_usd_total set 2026-08-04: local GPU inference, no API in the loop"
SKIP = ("row-socratic-segenc-test", "row-socratic-segenc-val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    changed = 0
    for p in sorted(ROWS.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rid = d.get("run_id", p.stem)
        if rid in SKIP or d.get("cost_usd_total") is not None:
            continue
        if rid in FRONTIER_COSTS:
            d["cost_usd_total"] = FRONTIER_COSTS[rid]
            note = FRONTIER_NOTE
        else:
            d["cost_usd_total"] = 0.0
            note = LOCAL_NOTE
        d["notes"] = (d.get("notes") or "").rstrip()
        d["notes"] = (d["notes"] + " | " + note) if d["notes"] else note
        print(f"{'APPLY' if a.apply else 'DRY  '} {rid}: cost_usd_total={d['cost_usd_total']}")
        if a.apply:
            p.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        changed += 1
    print(f"{changed} rows {'updated' if a.apply else 'would be updated'}")
    missing = set(FRONTIER_COSTS) - {p.stem for p in ROWS.glob("*.json")}
    if missing:
        raise SystemExit(f"frontier rows not found: {missing}")


if __name__ == "__main__":
    main()
