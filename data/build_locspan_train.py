"""Build locator-span fusion training data (M3 v9): the same 1095 specific train
queries as fusion.train.jsonl, prompts rebuilt from crossencoder locator spans at
the protocol budget (via pipeline.dump_prompts --split train), responses unchanged.
Closes the train/infer distribution gap exposed by M3 round 1 (oracle 0.3687 vs
pipeline 0.3352 on val specific queries)."""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    fusion = [json.loads(l) for l in
              (ROOT / "data" / "built" / "fusion.train.jsonl").open(encoding="utf-8")]
    prompts = {r["query_id"]: r["prompt"] for r in
               (json.loads(l) for l in
                (ROOT / "data" / "built" / "locspan.prompts.train.jsonl").open(encoding="utf-8"))}
    missing = {r["query_id"] for r in fusion} - set(prompts)
    if missing:
        raise SystemExit(f"{len(missing)} fusion query_ids missing locator prompts")
    out = ROOT / "data" / "built" / "fusion_locspan.train.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in fusion:
            f.write(json.dumps({"query_id": r["query_id"],
                                "prompt": prompts[r["query_id"]],
                                "response": r["response"]}) + "\n")
    print(f"wrote {out} ({len(fusion)} rows)")


if __name__ == "__main__":
    main()
