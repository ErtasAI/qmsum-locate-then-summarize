"""Contract tests for the shared peak-VRAM instrumentation.

These run on CPU. The point is not to check that CUDA reports memory, it is to pin the
conventions that make two systems' numbers comparable, and to pin the one behaviour that
protects the determinism guarantee: on a machine without CUDA the instrumentation must add
NOTHING to a prediction record, so a preds file written on CPU is byte-identical to what
the same code wrote before this module existed.
"""
import eval.vram as vram


def test_record_is_empty_without_cuda_so_preds_files_do_not_change():
    """A dict of Nones would still change every line of every preds file written on CPU.

    Returning {} means the instrumentation is invisible where it cannot measure, which is
    what lets it be merged into `run_pipeline.py` without reopening the byte-identical
    reproduction of the 281 test predictions.
    """
    if vram.available():
        rec = vram.record()
        assert set(rec) == {"peak_vram_gb", "peak_vram_reserved_gb"}
    else:
        assert vram.record() == {}


def test_reset_and_peak_are_safe_without_cuda():
    vram.reset()
    if not vram.available():
        assert vram.peak_gb() is None
        assert vram.peak_reserved_gb() is None


def test_summarize_reports_max_first_and_counts_only_measured_rows():
    s = vram.summarize([1.0, 2.0, None, 3.0, None])
    assert s["max"] == 3.0
    assert s["min"] == 1.0
    assert s["n"] == 3, "None rows are unmeasured, not zero, and must not enter the count"


def test_summarize_returns_none_when_nothing_was_measured():
    """A CPU run must not produce a VRAM row of zeros that reads like a measurement."""
    assert vram.summarize([None, None]) is None
    assert vram.summarize([]) is None


def test_median_of_an_even_sample_is_the_mean_of_the_middle_two():
    assert vram.summarize([1.0, 2.0, 3.0, 4.0])["median"] == 2.5


def test_median_of_an_odd_sample_is_the_middle_value():
    assert vram.summarize([1.0, 5.0, 9.0])["median"] == 5.0


def test_p95_tracks_the_tail_not_the_middle():
    """Memory here is driven by input length, whose distribution is long-tailed. A summary
    that only carried a median would hide the query that decides the card requirement."""
    vals = [1.0] * 90 + [9.0] * 10
    s = vram.summarize(vals)
    assert s["median"] == 1.0
    assert s["p95"] == 9.0
    assert s["max"] == 9.0


def test_p95_is_nearest_rank_on_the_sorted_index_and_the_definition_is_pinned():
    """Percentile definitions disagree, and a silent change of convention would move a
    published number without moving any code that looks numeric. This pins ours: nearest
    index on the 0-based sorted array, i.e. `sorted[round(0.95 * (n - 1))]`.

    The consequence to be aware of when reading the table: with 95 of 100 values equal,
    p95 returns that value rather than the tail. That is correct and is why `max` is the
    figure the paper leads with for a memory requirement.
    """
    vals = [1.0] * 95 + [9.0] * 5
    assert vram.summarize(vals)["p95"] == 1.0
    assert vram.summarize(vals)["max"] == 9.0


def test_p95_never_indexes_past_the_end_on_tiny_samples():
    for n in range(1, 6):
        s = vram.summarize([float(i) for i in range(n)])
        assert s["p95"] == float(n - 1)


def test_reserved_is_documented_as_the_larger_of_the_two():
    """Pinned as a docstring contract because the two are easy to confuse and the paper
    quotes them for different claims: allocated for architecture cost, reserved for what
    a card must hold."""
    assert "larger" in vram.peak_reserved_gb.__doc__
