"""M1 dataset: query + truncated full transcript -> reference. Train and val only."""
import json, pathlib
from data.normalize import load_meetings, load_queries
from eval import protocol

BUILT = pathlib.Path(__file__).resolve().parent / "built"

def main():
    for split in ("train", "val"):  # never test: training data must not see it
        meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
        rows = []
        for q in load_queries(split):
            transcript = protocol.truncate_words(
                protocol.format_transcript(meetings[q["meeting_id"]]), protocol.M1_TRUNCATE_WORDS)
            rows.append({"query_id": q["query_id"],
                         "prompt": protocol.PROMPT_TEMPLATE.format(query=q["query"], transcript=transcript),
                         "response": q["reference"]})
        with (BUILT / f"sft_m1.{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} examples")

if __name__ == "__main__":
    main()
