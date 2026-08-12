"""Contiguous utterance windows under a word budget."""
from eval import protocol

def windows(utterances, budget=None):
    budget = budget or protocol.CHUNK_WORDS
    out, cur, cur_words, start = [], [], 0, 0
    for i, u in enumerate(utterances):
        line = f"{u['speaker']}: {u['text']}"
        n = len(line.split())
        if cur and cur_words + n > budget:
            out.append({"start": start, "end": i - 1, "text": "\n".join(cur)})
            cur, cur_words, start = [], 0, i
        cur.append(line); cur_words += n
    if cur:
        out.append({"start": start, "end": len(utterances) - 1, "text": "\n".join(cur)})
    return out
