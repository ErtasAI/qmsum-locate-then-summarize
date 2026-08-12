"""Frozen scorer: ROUGE-1/2/L/Lsum (f-measure, stemmed) + optional BERTScore.

ROUGE-Lsum sentence handling, corrected 2026-07-27
--------------------------------------------------
`rouge_score` computes ROUGE-Lsum as a union-LCS over NEWLINE-SEPARATED sentences,
and silently degenerates to plain ROUGE-L when the text contains no newlines. Scoring
raw model output therefore measured whether a system happened to emit line breaks:
the Claude baselines format into paragraphs (68-100% of predictions contain "\n") and
gained up to +0.042 RLsum for it, while our model, all four community baselines and
the QMSum references contain none and gained nothing.

The fix is the standard summarization convention: derive sentence boundaries
ourselves and join on "\n", for BOTH candidate and reference, before scoring. The
load-bearing detail is that `sentences()` first flattens any newlines the system
already emitted, so every system is re-segmented by ONE rule. Without that flatten,
a system's own formatting would still leak into the metric and the fix would be
cosmetic.

A regex splitter rather than nltk.sent_tokenize: punkt is not present in this venv,
and downloading corpora at score time would make the frozen scorer depend on a
network fetch whose output changes between nltk releases. This rule is deterministic
and inspectable, which matters more here than handling every abbreviation edge case,
because it is applied identically to candidate and reference.

ROUGE-1/2/L are unaffected: the tokenizer splits on non-alphanumeric characters, so a
newline is just whitespace to them. `tests/test_scorer_lsum.py` asserts this rather
than assuming it.
"""
import re
from rouge_score import rouge_scorer

_ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)

# Split after ., ! or ? followed by whitespace. Semicolons and colons are not
# sentence boundaries and bullet lines without terminal punctuation stay as one
# segment, which is the conservative choice: it never invents boundaries that would
# inflate the union-LCS.
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def sentences(text):
    """Normalise text to one sentence per line for ROUGE-Lsum.

    Existing newlines are flattened FIRST so that a system that formats into
    paragraphs and one that emits a single line are segmented identically.
    """
    flat = " ".join((text or "").split())
    parts = [s.strip() for s in _SENT_END.split(flat) if s.strip()]
    return "\n".join(parts)


def score(predictions, references, with_bertscore=False):
    assert len(predictions) == len(references) and predictions
    totals = {k: 0.0 for k in ("rouge1", "rouge2", "rougeL", "rougeLsum")}
    for pred, ref in zip(predictions, references):
        r = _ROUGE.score(sentences(ref), sentences(pred))
        for k in totals:
            totals[k] += r[k].fmeasure
    out = {k: v / len(predictions) for k, v in totals.items()}
    out["bertscore_f1"] = None
    if with_bertscore:
        # BERTScore is embedding-based and sentence segmentation is irrelevant to it,
        # so it keeps the raw text.
        from bert_score import score as bs
        _, _, f1 = bs(predictions, references, lang="en", verbose=False)
        out["bertscore_f1"] = float(f1.mean())
    return out
