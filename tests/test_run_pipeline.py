from pipeline.run_pipeline import build_pipeline_prompt

def fake_locator(query, windows, budget_words):
    return windows[:1]

def oversize_locator(query, windows, budget_words):
    return [{"start": 0, "end": 0, "text": "spillover words " * 2500}]

def test_prompt_contains_located_text_only():
    utts = ([{"speaker": "A", "text": "relevant decision content " * 40}]
            + [{"speaker": "B", "text": "irrelevant chatter " * 40}] * 20)
    p = build_pipeline_prompt("What was decided?", utts, fake_locator)
    assert "relevant decision content" in p
    assert len(p.split()) <= 3400

def test_prompt_truncated_when_locator_overshoots():
    utts = [{"speaker": "A", "text": "anything " * 10}]
    p = build_pipeline_prompt("What was decided?", utts, oversize_locator)
    assert len(p.split()) <= 3400


# --- decode overrides for the density probe ---------------------------------

def test_no_overrides_reproduces_the_frozen_greedy_config():
    """The winner's 281 test predictions were verified byte-identical on a
    re-run. That guarantee is only worth having if the default path cannot
    drift, so this pins the exact kwargs rather than just 'no deviations'."""
    from pipeline.run_pipeline import build_gen_kwargs
    from eval import protocol
    kwargs, deviations = build_gen_kwargs()
    assert kwargs == {"max_new_tokens": protocol.MAX_NEW_TOKENS, "do_sample": False}
    assert deviations == {}


def test_each_override_is_recorded_as_a_deviation():
    from pipeline.run_pipeline import build_gen_kwargs
    kwargs, deviations = build_gen_kwargs(min_new_tokens=150, num_beams=4,
                                          length_penalty=1.5)
    assert kwargs["min_new_tokens"] == 150
    assert kwargs["num_beams"] == 4
    assert kwargs["length_penalty"] == 1.5
    assert deviations == {"min_new_tokens": 150, "num_beams": 4, "length_penalty": 1.5}


def test_socratic_decode_config_is_expressible_and_fully_recorded():
    """The exact published config we are now compared against: num_beams 4,
    no_repeat_ngram_size 3, early_stopping. Every part must land in kwargs AND be
    flagged, so a swept preds file can never pass as a frozen-protocol run."""
    from pipeline.run_pipeline import build_gen_kwargs
    kwargs, deviations = build_gen_kwargs(num_beams=4, no_repeat_ngram_size=3,
                                          early_stopping=True)
    assert kwargs["num_beams"] == 4
    assert kwargs["no_repeat_ngram_size"] == 3
    assert kwargs["early_stopping"] is True
    assert deviations == {"num_beams": 4, "no_repeat_ngram_size": 3, "early_stopping": True}


def test_new_decode_axes_default_to_absent_so_greedy_is_unchanged():
    """Adding sweep axes must not perturb the default path: the winner's 281 test
    predictions were verified byte-identical on a re-run under the default kwargs."""
    from pipeline.run_pipeline import build_gen_kwargs
    from eval import protocol
    kwargs, deviations = build_gen_kwargs()
    assert kwargs == {"max_new_tokens": protocol.MAX_NEW_TOKENS, "do_sample": False}
    assert deviations == {}
    assert "no_repeat_ngram_size" not in kwargs
    assert "early_stopping" not in kwargs


def test_a_zero_override_is_still_a_deviation_not_a_default():
    """0 is falsy; testing `if val` instead of `if val is not None` would drop it
    silently and the preds file would claim a protocol run."""
    from pipeline.run_pipeline import build_gen_kwargs
    _, deviations = build_gen_kwargs(min_new_tokens=0)
    assert deviations == {"min_new_tokens": 0}
