"""Raw QMSum jsonl -> normalized meetings + queries files."""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW, BUILT = HERE / "raw", HERE / "built"

def load_meetings(split):
    return [json.loads(l) for l in (BUILT / f"meetings.{split}.jsonl").open(encoding="utf-8")]

def load_queries(split):
    return [json.loads(l) for l in (BUILT / f"queries.{split}.jsonl").open(encoding="utf-8")]

def _spans(raw_spans):
    # raw: [["286", "480"], ...] utterance-index ranges as strings
    return [[int(a), int(b)] for a, b in (raw_spans or [])]

def main():
    BUILT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        meetings, queries = [], []
        raw = [json.loads(l) for l in (RAW / f"{split}.jsonl").open(encoding="utf-8")]
        for i, m in enumerate(raw):
            mid = f"{split}_{i}"
            utts = [{"speaker": u["speaker"], "text": u["content"]} for u in m["meeting_transcripts"]]
            meetings.append({"meeting_id": mid, "utterances": utts})
            for j, q in enumerate(m["general_query_list"]):
                queries.append({"query_id": f"{mid}_g{j}", "meeting_id": mid, "split": split,
                                "query_type": "general", "query": q["query"],
                                "reference": q["answer"], "gold_spans": []})
            for j, q in enumerate(m["specific_query_list"]):
                queries.append({"query_id": f"{mid}_s{j}", "meeting_id": mid, "split": split,
                                "query_type": "specific", "query": q["query"],
                                "reference": q["answer"],
                                "gold_spans": _spans(q.get("relevant_text_span"))})
        for name, rows in (("meetings", meetings), ("queries", queries)):
            with (BUILT / f"{name}.{split}.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(meetings)} meetings, {len(queries)} queries")

if __name__ == "__main__":
    main()
