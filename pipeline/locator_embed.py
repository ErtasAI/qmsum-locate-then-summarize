"""Embedding-retrieval locator baseline: rank windows by cosine similarity to the query."""
from sentence_transformers import SentenceTransformer, util

_MODEL = None
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME, device="cuda")
    return _MODEL

def select(query, windows, budget_words):
    m = _model()
    q = m.encode([query], normalize_embeddings=True)
    w = m.encode([x["text"] for x in windows], normalize_embeddings=True)
    ranked = sorted(zip(util.cos_sim(q, w)[0].tolist(), range(len(windows))), reverse=True)
    picked, used = [], 0
    for _, idx in ranked:
        n = len(windows[idx]["text"].split())
        if used + n > budget_words and picked:
            continue
        picked.append(idx); used += n
        if used >= budget_words:
            break
    return [windows[i] for i in sorted(picked)]
