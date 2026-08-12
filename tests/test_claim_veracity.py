"""This check is itself an LLM instrument, so the tests are mostly about it
refusing to report when its own control fails."""
import pytest

from eval import claim_veracity as V


def _key(**kw):
    return {cid: {"source": src, "claim": "c", "query_id": "q"} for cid, src in kw.items()}


def test_true_rate_counts_full_and_partial_as_true():
    """The whole question is whether an 'unsupported' claim is FALSE. A claim that
    is true but told incompletely is not false."""
    key = _key(a=V.OURS, b=V.OURS, c=V.OURS, d=V.OURS)
    verdicts = {"a": "SUPPORTED_FULL", "b": "SUPPORTED_PARTIAL",
                "c": "CONTRADICTED", "d": "NOT_IN_TRANSCRIPT"}
    res = V.aggregate(key, verdicts)
    assert res["rows"][V.OURS]["true_rate"] == pytest.approx(0.5)


def test_a_failed_gold_control_voids_the_whole_run():
    """If the verifier cannot confirm the reference's own facts against the
    transcript the reference summarises, its verdicts on our claims are noise."""
    key = _key(g1=V.GOLD, g2=V.GOLD, o1=V.OURS)
    verdicts = {"g1": "NOT_IN_TRANSCRIPT", "g2": "NOT_IN_TRANSCRIPT",
                "o1": "SUPPORTED_FULL"}
    res = V.aggregate(key, verdicts)
    assert res["control_passed"] is False
    assert res["verdict"].startswith("VOID")


def test_a_passing_gold_control_makes_the_run_usable():
    key = _key(**{f"g{i}": V.GOLD for i in range(10)})
    verdicts = {f"g{i}": ("SUPPORTED_FULL" if i < 9 else "NOT_IN_TRANSCRIPT")
                for i in range(10)}
    res = V.aggregate(key, verdicts)
    assert res["rows"][V.GOLD]["true_rate"] == pytest.approx(0.9)
    assert res["control_passed"] is True


def test_unlabelled_claims_are_reported_not_silently_dropped():
    key = _key(a=V.OURS, b=V.OURS)
    res = V.aggregate(key, {"a": "SUPPORTED_FULL", "b": "NONSENSE_LABEL"})
    assert res["rows"][V.OURS]["unlabelled"] == 1


def test_claims_with_no_verdict_do_not_count_toward_n():
    key = _key(a=V.OURS, b=V.OURS)
    res = V.aggregate(key, {"a": "SUPPORTED_FULL"})
    assert res["rows"][V.OURS]["n"] == 1


# --- the filename collision that actually fired ------------------------------

def test_fact_decompositions_of_different_sizes_get_different_filenames():
    """facts__gpt-5.6-sol.json held n=60 because the n=281 run wrote the same
    path. Found when a control that should have drawn ~118 facts drew 13."""
    from eval import judge_facts as F
    assert F.facts_tag("m", n=60) == "facts__m"
    assert F.facts_tag("m", n=281) == "facts__m__n281"
    assert F.facts_tag("m", n=281) != F.facts_tag("m", n=60)


def test_facts_tag_still_keys_on_split_as_well():
    from eval import judge_facts as F
    assert F.facts_tag("m", "val", 281) == "facts__m__n281__val"
