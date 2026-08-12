"""Stage targets via greedy ROUGE alignment (the Summ^N method, provenance: train split only).

fusion.*: gold-located spans + query -> full reference (stage-3 training data)
coarse.*: each gold-relevant window + query -> reference sentences aligned to it
Alignment scores rouge1 recall with the sentence as target (Summ^N semantics); argument order matters.
"""
import json, pathlib, re
from rouge_score import rouge_scorer
from data.normalize import load_meetings, load_queries
from pipeline.chunker import windows
from data.build_locator_data import overlaps
from eval import protocol

BUILT = pathlib.Path(__file__).resolve().parent / "built"
_R1 = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

def best_window(sentence, window_texts):
    """Index of the window capturing the sentence best: rouge1 recall, sentence as target."""
    return max(range(len(window_texts)), key=lambda k: _R1.score(sentence, window_texts[k])["rouge1"].recall)

def main():
    for split in ("train", "val"):
        meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
        fusion, coarse = [], []
        for q in load_queries(split):
            if q["query_type"] != "specific" or not q["gold_spans"]:
                continue
            w = windows(meetings[q["meeting_id"]])
            gold = [x for x in w if overlaps(x, q["gold_spans"])]
            if not gold:
                continue
            span_text = protocol.truncate_words(
                "\n".join(g["text"] for g in gold), protocol.SPAN_BUDGET_WORDS)
            fusion.append({"query_id": q["query_id"],
                           "prompt": protocol.PROMPT_TEMPLATE.format(query=q["query"], transcript=span_text),
                           "response": q["reference"]})
            if len(gold) > 1:  # coarse stage only matters for multi-window spans
                for gi, g in enumerate(gold):
                    aligned = [s for s in sentences(q["reference"])
                               if best_window(s, [g["text"] for g in gold]) == gi]
                    if aligned:
                        coarse.append({"query_id": f"{q['query_id']}_w{gi}",
                                       "prompt": protocol.PROMPT_TEMPLATE.format(
                                           query=q["query"], transcript=g["text"]),
                                       "response": " ".join(aligned)})
        for name, rows in (("fusion", fusion), ("coarse", coarse)):
            with (BUILT / f"{name}.{split}.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{name}.{split}: {len(rows)}")

if __name__ == "__main__":
    main()
