"""Span-regime training data for SegEnc, in the (query, source, target) form its chunker needs.

WHY A SEPARATE BUILDER. `fusion.*.jsonl` stores a single pre-formatted `prompt` string with the
query already interpolated into it. SegEnc cannot use that: its `ChunkTokenizer` prepends
`<s>{query}</s>` to EVERY chunk, so it needs the query and the source text as separate fields.

THE DESIGN CHOICE THIS FILE ENCODES. The question is whether a Fusion-in-Decoder model can work in
a retrieval regime, so the comparison against our own 1.2B has to vary the ARCHITECTURE and hold
the DATA RECIPE fixed. So the source here is byte-identical to what our own summarizer trains on:
gold-overlapping 900-word windows, joined in transcript order, truncated to `SPAN_BUDGET_WORDS`,
specific queries only. Median about 1,716 words.

That means SegEnc inherits our train/inference asymmetry too: trained on gold-overlapping windows,
evaluated on locator-packed spans at a 2,000-word budget. That asymmetry is deliberate in our own
recipe (the paper's data-recipe section, and the locspan retrain that made it worse), and mirroring
it is what keeps the comparison fair. Training SegEnc on located spans instead would change two
things at once.

THE CONTROL. The source text is re-derived from the normalised data rather than parsed out of the
existing prompt strings, and then **verified byte-for-byte against `fusion.*.jsonl`**. Parsing the
prompts would have been fewer lines and would silently drift if the template ever changed; deriving
plus verifying catches a mismatch instead of inheriting one. `--verify-only` runs the check alone.
"""
import argparse
import json
import pathlib

from data.build_locator_data import overlaps
from data.normalize import load_meetings, load_queries
from eval import protocol
from pipeline.chunker import windows

BUILT = pathlib.Path(__file__).resolve().parent / "built"


def gold_span_source(utterances, gold_spans):
    """Gold-overlapping windows at the protocol chunk size, truncated to the span budget.

    Mirrors data/build_stage_targets.py exactly. Note these are gold-overlapping WINDOWS, not the
    precise gold utterance ranges, so the text is coarser and longer than the annotations alone
    (median about 1,716 words against 482 for the raw annotated spans).
    """
    w = windows(utterances)
    gold = [x for x in w if overlaps(x, gold_spans)]
    if not gold:
        return None
    return protocol.truncate_words("\n".join(g["text"] for g in gold),
                                   protocol.SPAN_BUDGET_WORDS)


def build(split):
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
    rows = []
    for q in load_queries(split):
        if q["query_type"] != "specific" or not q["gold_spans"]:
            continue
        source = gold_span_source(meetings[q["meeting_id"]], q["gold_spans"])
        if source is None:
            continue
        rows.append({"query_id": q["query_id"], "query": q["query"],
                     "source": source, "target": q["reference"]})
    return rows


def verify_against_fusion(split, rows):
    """Assert our source text is byte-identical to the transcript inside fusion.<split>.jsonl.

    If this fails, the two recipes have diverged and any architecture comparison built on them is
    confounded. Returns (n_checked, list of mismatched query_ids).
    """
    path = BUILT / f"fusion.{split}.jsonl"
    if not path.exists():
        return 0, [f"missing {path.name}"]
    fusion = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            fusion[r["query_id"]] = r
    mismatched = []
    for r in rows:
        f = fusion.get(r["query_id"])
        if f is None:
            mismatched.append(f"{r['query_id']} absent from fusion")
            continue
        expected = protocol.PROMPT_TEMPLATE.format(query=r["query"], transcript=r["source"])
        if f["prompt"] != expected:
            mismatched.append(r["query_id"])
        if f["response"] != r["target"]:
            mismatched.append(f"{r['query_id']} target")
    return len(rows), mismatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    for split in ("train", "val"):
        rows = build(split)
        n, bad = verify_against_fusion(split, rows)
        status = "OK" if not bad else f"MISMATCH ({len(bad)}): {bad[:3]}"
        words = sorted(len(r["source"].split()) for r in rows)
        median = words[len(words) // 2] if words else 0
        print(f"{split}: {len(rows)} rows, median source {median} words | "
              f"verify vs fusion.{split}: {status}")
        if bad:
            raise SystemExit("source text diverges from the recipe our own summarizer trained on; "
                             "an architecture comparison built on this would be confounded")
        if not a.verify_only:
            out = BUILT / f"segenc-spans.{split}.jsonl"
            with out.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
