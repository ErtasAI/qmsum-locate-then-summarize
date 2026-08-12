"""The frozen protocol is frozen. This test is the enforcement mechanism.

Edward's decision, 2026-07-30: the promoted system configuration deviates from the protocol on
two locator-side values (`CHUNK_WORDS` 900 -> 375, `SPAN_BUDGET_WORDS` 3,000 -> 2,000), and we
**report the deviation rather than amending the protocol**. A protocol that gets edited whenever it
constrains a headline was never frozen, and the rest of the paper leans on the claim that these
values were fixed before any comparison was run.

The pressure to "tidy up" protocol.py so it matches the reported system is exactly the thing this
guards against, and it will look like a harmless cleanup to whoever does it. Every value below is
baked into cached rows, cached API responses, and published numbers.

If a value here genuinely must change, that is a new protocol version and a deliberate decision:
update this test, record the change and the reason in the changelog, and state which rows were
regenerated. Do not silently edit protocol.py to make this test pass.
"""
from eval import protocol


def test_locator_side_values_are_unchanged():
    """The two the promoted config deviates from. These are the ones under pressure."""
    assert protocol.CHUNK_WORDS == 900
    assert protocol.SPAN_BUDGET_WORDS == 3000


def test_decoding_and_seed_are_unchanged():
    assert protocol.SEED == 20260723
    assert protocol.TEMPERATURE == 0.0
    assert protocol.MAX_NEW_TOKENS == 512


def test_input_budgets_are_unchanged():
    assert protocol.M1_TRUNCATE_WORDS == 4500
    # Raised 14000 -> 40000 on 2026-07-27 BEFORE any baseline ran, so no cached row was
    # invalidated. It exceeds the longest transcript in either split, so the cap bites 0/281.
    assert protocol.PROMPT_WORD_BUDGET == 40000


def test_judge_sample_size_is_unchanged():
    assert protocol.JUDGE_SAMPLE_N == 60


def test_prompt_template_is_unchanged():
    """The baselines' cached responses were generated against this exact string, and reruns
    resume from cache keyed on the query rather than on the prompt, so a change here would
    silently mix two prompt formats inside one row."""
    assert protocol.PROMPT_TEMPLATE == (
        "You are given a meeting transcript and a query. Answer the query with a "
        "concise summary based only on the transcript.\n\n"
        "Query: {query}\n\nTranscript:\n{transcript}\n\nSummary:"
    )
