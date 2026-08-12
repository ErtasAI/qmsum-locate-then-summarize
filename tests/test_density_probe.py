"""The screening exists to stop a bad run reaching a paid judge, so the tests
are about it saying NO for the right reasons."""
import pytest

from eval import density_probe as D


def test_repeat_4gram_is_zero_on_varied_text_and_high_on_a_loop():
    assert D.repeat_4gram("the team agreed to ship the product on friday afternoon") == 0.0
    looped = "the remote should be light " * 6
    assert D.repeat_4gram(looped) > D.DEGENERATE_REPEAT_4GRAM


def test_short_text_cannot_be_degenerate_by_arithmetic():
    """Fewer than four words yields no 4-grams; a divide would raise."""
    assert D.repeat_4gram("too short") == 0.0


def test_a_repetitive_variant_is_blocked_from_marking():
    base = {"n": 100, "words_median": 50, "repeat_4gram_mean": 0.01, "degenerate_rows": 0}
    var = {"n": 100, "words_median": 120, "repeat_4gram_mean": 0.30, "degenerate_rows": 40}
    v = D.verdict(base, var)
    assert v["worth_marking"] is False
    assert len(v["blocked_because"]) >= 1


def test_a_variant_that_did_not_actually_get_longer_is_blocked():
    """Marking it would spend money to answer a question it does not pose."""
    base = {"n": 100, "words_median": 50, "repeat_4gram_mean": 0.01, "degenerate_rows": 0}
    var = {"n": 100, "words_median": 52, "repeat_4gram_mean": 0.01, "degenerate_rows": 0}
    v = D.verdict(base, var)
    assert v["worth_marking"] is False
    assert "length barely moved" in v["blocked_because"][0]


def test_a_longer_clean_variant_is_approved():
    base = {"n": 100, "words_median": 50, "repeat_4gram_mean": 0.01, "degenerate_rows": 0}
    var = {"n": 100, "words_median": 110, "repeat_4gram_mean": 0.015, "degenerate_rows": 2}
    assert D.verdict(base, var)["worth_marking"] is True


def test_a_pristine_baseline_does_not_make_every_variant_look_repetitive():
    """A baseline at exactly 0.0 repeat would make any nonzero variant 'more than
    double' it, blocking clean runs. The floor in the comparison prevents that."""
    base = {"n": 100, "words_median": 50, "repeat_4gram_mean": 0.0, "degenerate_rows": 0}
    var = {"n": 100, "words_median": 110, "repeat_4gram_mean": 0.008, "degenerate_rows": 0}
    assert D.verdict(base, var)["worth_marking"] is True


def test_profile_reports_length_against_the_reference():
    preds = {"a": "one two three four five six", "b": "one two three"}
    refs = {"a": "one two three", "b": "one two three"}
    p = D.profile(preds, refs)
    assert p["length_vs_reference"] == pytest.approx(1.5)
