"""Query-window pairs labeled by gold-span overlap. Specific queries only.

`--chunk-words` added 2026-07-30. The default path is unchanged and still writes
`locator.{split}.jsonl` at `protocol.CHUNK_WORDS`, because that 13,738-pair file at 900 words is
cited in the paper's system section. **A non-default size writes a size-suffixed filename**
(`locator.{split}.w{N}.jsonl`) so a variant build can never silently replace the cited one. Two
collisions of exactly this kind cost real debugging time in session 5.

Why a size other than 900 is worth building: the cross-encoder's context is 512 tokens while a
900-word window tokenizes to a median 1,150, so 97.3% of windows are truncated and the scorer sees
a mean 47.7% of each. Documented in the paper's locator truncation analysis.
"""
import argparse, json, pathlib
from data.normalize import load_meetings, load_queries
from eval import protocol
from pipeline.chunker import windows

BUILT = pathlib.Path(__file__).resolve().parent / "built"

def overlaps(window, spans):
    return any(not (window["end"] < s or window["start"] > e) for s, e in spans)

def path_for(split, chunk_words):
    """Default size keeps the historical filename; any other size is suffixed."""
    if chunk_words == protocol.CHUNK_WORDS:
        return BUILT / f"locator.{split}.jsonl"
    return BUILT / f"locator.{split}.w{chunk_words}.jsonl"

def build(split, chunk_words):
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
    win_cache = {mid: windows(u, chunk_words) for mid, u in meetings.items()}
    rows = []
    for q in load_queries(split):
        if q["query_type"] != "specific" or not q["gold_spans"]:
            continue
        for w in win_cache[q["meeting_id"]]:
            rows.append({"query": q["query"], "text": w["text"],
                         "label": int(overlaps(w, q["gold_spans"])),
                         "query_id": q["query_id"], "start": w["start"], "end": w["end"]})
    out = path_for(split, chunk_words)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = sum(r["label"] for r in rows)
    print(f"{split}: {len(rows)} pairs, {pos} positive ({pos/len(rows):.1%}) -> {out.name}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-words", type=int, default=protocol.CHUNK_WORDS,
                    help=f"window size in words (default {protocol.CHUNK_WORDS}, the frozen "
                         f"protocol value). A non-default size writes a suffixed file.")
    a = ap.parse_args()
    for split in ("train", "val"):
        build(split, a.chunk_words)

if __name__ == "__main__":
    main()
