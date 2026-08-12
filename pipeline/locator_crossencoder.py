"""Cross-encoder locator: fine-tuned reranker over query-window pairs."""
import json, pathlib, argparse
from sentence_transformers import CrossEncoder, InputExample

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CKPT = ROOT / "checkpoints" / "locator-crossencoder"
_MODEL = None

def train(epochs=2, batch_size=32, train_file=None):
    import torch.utils.data as tud
    from transformers import set_seed
    from eval import protocol
    set_seed(protocol.SEED)  # seeds python/numpy/torch incl. the DataLoader shuffle
    train_file = train_file or (ROOT / "data" / "built" / "locator.train.jsonl")
    rows = [json.loads(l) for l in pathlib.Path(train_file).open(encoding="utf-8")]
    examples = [InputExample(texts=[r["query"], r["text"]], label=float(r["label"])) for r in rows]
    model = CrossEncoder(BASE, num_labels=1, device="cuda")
    model.fit(train_dataloader=tud.DataLoader(examples, shuffle=True, batch_size=batch_size),
              epochs=epochs, warmup_steps=100, output_path=str(CKPT))
    # Adaptation (sentence-transformers 3.4.1): CrossEncoder.fit() only calls
    # save() via its internal eval callback, which only fires when an
    # `evaluator` is passed. With no evaluator (as speced: binary relevance,
    # no held-out eval loop), fit() creates output_path but never writes the
    # model into it. Save explicitly so the checkpoint persists.
    model.save(str(CKPT))
    print(f"saved {CKPT}")

def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder(str(CKPT), device="cuda")
    return _MODEL

def select(query, windows, budget_words):
    scores = _model().predict([(query, w["text"]) for w in windows])
    ranked = sorted(zip(scores, range(len(windows))), reverse=True)
    picked, used = [], 0
    for _, idx in ranked:
        n = len(windows[idx]["text"].split())
        if used + n > budget_words and picked:
            continue
        picked.append(idx); used += n
        if used >= budget_words:
            break
    return [windows[i] for i in sorted(picked)]

def recall_at_budget(select_fn, split, budget, chunk_words=None):
    """Fraction of gold-positive windows captured by the selection, macro over queries.

    `chunk_words` defaults to the frozen protocol value. Pass it to evaluate a locator that
    was trained at a different window size; it must match the size the locator was trained on,
    or the model sees a text length it never saw in training.
    """
    from data.normalize import load_meetings, load_queries
    from pipeline.chunker import windows as make_windows
    from data.build_locator_data import overlaps
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
    recalls = []
    for q in load_queries(split):
        if q["query_type"] != "specific" or not q["gold_spans"]:
            continue
        w = make_windows(meetings[q["meeting_id"]], chunk_words)
        gold = [x for x in w if overlaps(x, q["gold_spans"])]
        if not gold:
            continue
        picked = select_fn(q["query"], w, budget)
        got = sum(1 for g in gold if any(p["start"] == g["start"] for p in picked))
        recalls.append(got / len(gold))
    if not recalls:
        raise SystemExit("recall_at_budget: no gold-positive queries in split; nothing to score")
    return sum(recalls) / len(recalls)

def gold_utterance_recall(select_fn, split, budget, chunk_words=None):
    """Fraction of GOLD UTTERANCES present in the packed selection, macro over queries.

    Added 2026-07-30, and it is the metric to use whenever window size varies.
    `recall_at_budget` counts gold-positive WINDOWS, so both its numerator and denominator
    move with `chunk_words`: at 900 words a query's gold content touches a median of 2 windows
    out of 12, and at 375 words it touches more windows out of more, with more slots available.
    Those two recalls are different quantities and comparing them would be meaningless.

    This counts utterances, which the window size does not change, so it is comparable across
    granularities and is also closer to what the summarizer actually needs: whether the gold
    content reached the prompt at all.
    """
    from data.normalize import load_meetings, load_queries
    from pipeline.chunker import windows as make_windows
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
    recalls = []
    for q in load_queries(split):
        if q["query_type"] != "specific" or not q["gold_spans"]:
            continue
        utts = meetings[q["meeting_id"]]
        gold = set()
        for s, e in q["gold_spans"]:
            gold.update(range(s, min(e, len(utts) - 1) + 1))
        if not gold:
            continue
        w = make_windows(utts, chunk_words)
        picked = select_fn(q["query"], w, budget)
        got = set()
        for p in picked:
            got.update(range(p["start"], p["end"] + 1))
        recalls.append(len(gold & got) / len(gold))
    if not recalls:
        raise SystemExit("gold_utterance_recall: no gold-positive queries in split")
    return sum(recalls) / len(recalls)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "eval"])
    a = ap.parse_args()
    if a.command == "train":
        train()
    else:
        from pipeline import locator_embed
        for name, fn in (("embed", locator_embed.select), ("crossencoder", select)):
            for budget in (2000, 3000, 4000):
                r = recall_at_budget(fn, "val", budget)
                print(f"{name}: recall@{budget}w = {r:.3f}")
