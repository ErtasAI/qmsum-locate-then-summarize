"""ROUGE-Lsum sentence segmentation: the property is FORMAT INVARIANCE.

Two systems that say the same thing must score the same, whether one emits a single
line and the other emits paragraphs. Before 2026-07-27 they did not, and the Claude
baselines gained up to +0.042 RLsum for formatting alone.
"""
import pytest
from eval.scorer import score, sentences

ONE_LINE = "The committee agreed to fund the trial. Members raised concerns about cost. A vote was deferred."
PARAGRAPHS = "The committee agreed to fund the trial.\n\nMembers raised concerns about cost.\nA vote was deferred."
REFERENCE = "The committee agreed to fund the trial and members raised cost concerns, so a vote was deferred."


def test_sentences_splits_on_terminal_punctuation():
    assert sentences("One. Two! Three?") == "One.\nTwo!\nThree?"


def test_sentences_flattens_the_systems_own_newlines_first():
    """The load-bearing behaviour. A system that already emits newlines must be
    re-segmented by the same rule as one that does not, or its formatting still
    leaks into the metric."""
    assert sentences(ONE_LINE) == sentences(PARAGRAPHS)


def test_identical_content_scores_identically_regardless_of_formatting():
    """The bug this file exists to prevent: newline-formatted output outscoring
    identical single-line output."""
    a = score([ONE_LINE], [REFERENCE])
    b = score([PARAGRAPHS], [REFERENCE])
    assert a == b


def test_lsum_is_now_a_real_measurement_distinct_from_l():
    """With segmentation applied, RLsum is a union-LCS computed per reference
    sentence, so unlike RL it is not penalised for sentence ORDER. A candidate that
    states the same two facts in the opposite order should lose on RL but hold up on
    RLsum. Before the fix RLsum silently degenerated to RL on our own output, which
    is why every pre-fix row had them equal to 16 decimals.

    Note this needs a multi-sentence REFERENCE: with a single-sentence reference the
    union collapses and the two metrics can coincide by construction.
    """
    ref = "The budget was approved. The chair resigned."
    reordered = "The chair resigned. The budget was approved."
    s = score([reordered], [ref])
    assert s["rougeLsum"] > s["rougeL"]


@pytest.mark.parametrize("text", [ONE_LINE, PARAGRAPHS, "Single sentence only."])
def test_rouge_1_2_l_are_unaffected_by_segmentation(text):
    """Newlines are whitespace to the n-gram tokenizer, so the headline metrics and
    every model-selection decision made under the old scorer stand unchanged.
    Asserted rather than assumed."""
    from rouge_score import rouge_scorer
    r = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    raw = r.score(REFERENCE, text)
    seg = r.score(sentences(REFERENCE), sentences(text))
    for k in ("rouge1", "rouge2", "rougeL"):
        assert abs(raw[k].fmeasure - seg[k].fmeasure) < 1e-12


def test_empty_and_none_predictions_do_not_crash():
    assert sentences("") == "" and sentences(None) == ""
    score(["", "x."], ["a b c.", "x."])
