"""Tests for aligning a third party's published predictions to our split.

The failure this module exists to prevent is a SILENT one: lining up two files of
different length by index shifts every pair after the gap and still yields a plausible
ROUGE. So the tests concentrate on (a) the gap being located in the right place and
(b) the controls actually failing when the alignment is junk.
"""
import pytest

from eval import external_preds as ep


def test_token_f1_identical_is_one():
    t = ep.tokens("the cat sat on the mat")
    assert ep.token_f1(t, t) == pytest.approx(1.0)


def test_token_f1_disjoint_is_zero():
    assert ep.token_f1(ep.tokens("alpha beta"), ep.tokens("gamma delta")) == 0.0


def test_token_f1_handles_empty():
    assert ep.token_f1([], ep.tokens("anything")) == 0.0


def test_align_recovers_a_dropped_reference_in_the_middle():
    refs = ["alpha alpha", "beta beta", "gamma gamma", "delta delta"]
    preds = ["alpha alpha", "gamma gamma", "delta delta"]  # reference index 1 dropped
    sim = ep.similarity_matrix(preds, refs)
    assert ep.align(sim) == [0, 2, 3]


def test_align_recovers_a_dropped_reference_at_the_end():
    refs = ["alpha alpha", "beta beta", "gamma gamma"]
    preds = ["alpha alpha", "beta beta"]
    sim = ep.similarity_matrix(preds, refs)
    assert ep.align(sim) == [0, 1]


def test_align_recovers_a_dropped_reference_at_the_start():
    refs = ["alpha alpha", "beta beta", "gamma gamma"]
    preds = ["beta beta", "gamma gamma"]
    sim = ep.similarity_matrix(preds, refs)
    assert ep.align(sim) == [1, 2]


def test_align_output_is_strictly_monotone():
    refs = [f"token{i} shared" for i in range(12)]
    preds = [refs[i] for i in range(12) if i not in (3, 7)]
    mapping = ep.align(ep.similarity_matrix(preds, refs))
    assert mapping == sorted(set(mapping))
    assert len(mapping) == len(preds)


def test_align_rejects_more_predictions_than_references():
    sim = [[1.0], [1.0]]
    with pytest.raises(ValueError):
        ep.align(sim)


def test_align_handles_equal_lengths_as_the_identity():
    refs = ["alpha alpha", "beta beta", "gamma gamma"]
    sim = ep.similarity_matrix(refs, refs)
    assert ep.align(sim) == [0, 1, 2]


def test_verify_passes_on_a_clean_alignment():
    refs = [f"unique{i} word{i} filler{i}" for i in range(10)]
    preds = [refs[i] for i in range(10) if i != 4]
    sim = ep.similarity_matrix(preds, refs)
    checks = ep.verify(sim, ep.align(sim))
    assert checks["passed"]
    assert checks["monotone"]
    assert checks["top1_recovery"] == pytest.approx(1.0)
    assert checks["slack"] == 1


def test_verify_does_not_require_a_rigid_margin_at_zero_slack():
    # At equal lengths the identity is the ONLY monotone alignment, so the best rigid
    # offset is that same alignment and margin_over_rigid is necessarily 1.0. Requiring a
    # margin there would make the control unsatisfiable, which is the bug this asserts is
    # fixed.
    refs = [f"unique{i} word{i} filler{i}" for i in range(10)]
    sim = ep.similarity_matrix(refs, refs)
    checks = ep.verify(sim, ep.align(sim))
    assert checks["slack"] == 0
    assert checks["margin_over_rigid"] is None
    assert checks["passed"]


def test_verify_still_requires_a_rigid_margin_when_there_is_slack():
    refs = [f"unique{i} word{i} filler{i}" for i in range(10)]
    preds = [refs[i] for i in range(10) if i != 4]
    checks = ep.verify(*(lambda s: (s, ep.align(s)))(ep.similarity_matrix(preds, refs)))
    assert checks["slack"] == 1
    assert checks["margin_over_rigid"] is not None


def test_verify_reports_top1_lift_relative_to_chance():
    refs = [f"unique{i} word{i} filler{i}" for i in range(10)]
    sim = ep.similarity_matrix(refs, refs)
    checks = ep.verify(sim, ep.align(sim))
    # Perfect recovery at n=10 is a 10x lift over a 1-in-10 chance.
    assert checks["top1_lift_over_chance"] == pytest.approx(10.0)


def test_verify_fails_when_matched_pairs_are_no_better_than_unmatched():
    # Every text identical: matched pairs are indistinguishable from unmatched ones, so
    # the alignment carries no information and must not pass.
    refs = ["the same text everywhere"] * 8
    sim = ep.similarity_matrix(refs, refs)
    checks = ep.verify(sim, ep.align(sim))
    assert checks["mean_matched_percentile"] == pytest.approx(0.0)
    assert not checks["passed"]


def test_percentile_control_is_immune_to_a_shared_vocabulary_floor():
    # The case that broke the magnitude ratio: every pair shares heavy boilerplate, so
    # unmatched similarity is high in absolute terms, but the matched reference still ranks
    # top for every prediction. A correct alignment must pass here.
    boiler = "the committee discussed the meeting and agreed on the following points about"
    refs = [f"{boiler} distinctive{i} subject{i} matter{i}" for i in range(12)]
    sim = ep.similarity_matrix(refs, refs)
    checks = ep.verify(sim, ep.align(sim))
    assert checks["mean_unmatched_similarity"] > 0.5      # the floor is real
    assert checks["mean_matched_percentile"] == pytest.approx(1.0)
    assert checks["passed"]


def test_verify_rejects_a_shifted_alignment_on_realistic_prose():
    # The silent failure the module exists to catch: same texts, off by one. The forced
    # mapping is wrong, and the percentile control must notice.
    boiler = "the committee discussed the meeting and agreed on the following points about"
    refs = [f"{boiler} distinctive{i} subject{i} matter{i}" for i in range(12)]
    sim = ep.similarity_matrix(refs, refs)
    shifted = [(i + 1) % len(refs) for i in range(len(refs))]
    shifted.sort()  # keep it monotone so only the correspondence is wrong
    checks = ep.verify(sim, list(range(len(refs))))
    assert checks["passed"]
    off_by_one = ep.verify(ep.similarity_matrix(refs[1:], refs), list(range(len(refs) - 1)))
    assert not off_by_one["passed"]


def test_verify_fails_when_predictions_are_unrelated_to_references():
    refs = [f"reference{i} content{i}" for i in range(10)]
    preds = [f"totally different{i} text{i}" for i in range(9)]
    sim = ep.similarity_matrix(preds, refs)
    checks = ep.verify(sim, ep.align(sim))
    assert not checks["passed"]


def test_best_rigid_offset_is_the_null_model_the_dp_must_beat():
    # Reference 0 dropped, so a rigid offset of 1 IS the correct alignment here and the
    # DP should not claim a margin over it. Guards against the margin control being
    # trivially satisfiable.
    refs = [f"unique{i} word{i}" for i in range(6)]
    preds = [refs[i] for i in range(1, 6)]
    sim = ep.similarity_matrix(preds, refs)
    rigid = ep.best_rigid_offset(sim)
    matched = [sim[i][ep.align(sim)[i]] for i in range(len(preds))]
    assert rigid == pytest.approx(sum(matched) / len(matched))


def test_load_predictions_drops_blank_lines(tmp_path):
    p = tmp_path / "preds.txt"
    p.write_text("first line\n\nsecond line\n\n", encoding="utf-8")
    assert ep.load_predictions(p) == ["first line", "second line"]


def test_controls_are_fixed_constants_not_computed():
    # These thresholds are the whole point of the module. A future edit that makes them
    # data-dependent, or that quietly relaxes one to admit a row, should fail here.
    assert ep.MIN_MEAN_SIM == 0.20
    assert ep.MIN_MEAN_PERCENTILE == 0.90
    assert ep.MIN_TOP1_LIFT == 10.0
    assert ep.MIN_MARGIN == 1.25
    # The one that actually gates a published row; the four above are only
    # sanity checks.
    assert ep.ROUGE1_TOLERANCE == 3.0
