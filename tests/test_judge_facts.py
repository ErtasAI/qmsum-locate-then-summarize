"""Fact-level scoring. These numbers are meant to be the trustworthy arm of M4
after pairwise judging failed its control, so the arithmetic and the fairness
properties are both pinned here."""
import json
import pytest
from eval import judge_facts as F
from eval import run_judge_panel as P


# --- prompt construction and fairness properties ----------------------------

def test_extract_prompt_never_shows_a_system_summary():
    """The fact list must be a property of the reference alone. If a contestant's
    text could reach the extractor, the target could be shaped by a contestant."""
    p = F.build_extract_prompt("Why a clip?", "The PM suggested a clip.")
    assert "Summary" not in p
    assert "The PM suggested a clip." in p


def test_mark_prompt_is_blind_and_pointwise():
    p = F.build_mark_prompt("Why a clip?", ["PM suggested a clip"], "A summary here.")
    for ident in ["gpt", "openai", "claude", "anthropic", "lfm", "qwen", "ertas", "baseline"]:
        assert ident not in p.lower()
    assert "Summary B" not in p          # pointwise: only one summary, so no ordering
    assert "1. PM suggested a clip" in p


def test_mark_prompt_asks_for_paraphrase_credit_and_unsupported_claims():
    p = F.build_mark_prompt("q", ["f1"], "s")
    assert "paraphrase" in p
    assert "NOT supported" in p


# --- JSON parsing -----------------------------------------------------------

def test_parses_a_bare_json_object():
    assert F.parse_json_block('{"facts": ["a"]}') == {"facts": ["a"]}


def test_parses_json_wrapped_in_prose_or_fences():
    assert F.parse_json_block('Sure!\n```json\n{"facts": ["a"]}\n```\nDone') == {"facts": ["a"]}


def test_unparseable_reply_is_none_not_an_empty_default():
    """An empty default would score as a summary that covered nothing, which is
    a silent wrong answer rather than a visible failure."""
    assert F.parse_json_block("I could not do that") is None
    assert F.parse_json_block("") is None
    assert F.parse_json_block('{"facts": [broken}') is None


def test_non_object_json_is_rejected():
    assert F.parse_json_block('[1, 2, 3]') is None


# --- scoring arithmetic -----------------------------------------------------

def test_full_coverage_and_full_support_scores_one():
    s = F.score_marking({"covered_fact_numbers": [1, 2], "summary_claims": ["a", "b"],
                         "unsupported_claim_numbers": []}, n_facts=2)
    assert s["recall"] == 1.0 and s["precision"] == 1.0


def test_padding_lowers_precision_and_leaves_recall_alone():
    """The property the pairwise judge lacked: length cannot buy score."""
    tight = F.score_marking({"covered_fact_numbers": [1, 2], "summary_claims": ["a", "b"],
                             "unsupported_claim_numbers": []}, n_facts=2)
    padded = F.score_marking({"covered_fact_numbers": [1, 2],
                              "summary_claims": ["a", "b", "x", "y", "z"],
                              "unsupported_claim_numbers": [3, 4, 5]}, n_facts=2)
    assert padded["recall"] == tight["recall"]
    assert padded["precision"] < tight["precision"]
    assert padded["precision"] == pytest.approx(2 / 5)


def test_out_of_range_fact_numbers_are_discarded():
    """A judge that invents fact 9 of 2 must not score above 1.0 recall."""
    s = F.score_marking({"covered_fact_numbers": [1, 9, 0, -3], "summary_claims": ["a"],
                         "unsupported_claim_numbers": []}, n_facts=2)
    assert s["n_covered"] == 1 and s["recall"] == 0.5


def test_duplicate_fact_numbers_count_once():
    s = F.score_marking({"covered_fact_numbers": [1, 1, 1], "summary_claims": ["a"],
                         "unsupported_claim_numbers": []}, n_facts=2)
    assert s["n_covered"] == 1


def test_numeric_strings_from_the_judge_are_accepted():
    s = F.score_marking({"covered_fact_numbers": ["1", "2"], "summary_claims": ["a"],
                         "unsupported_claim_numbers": []}, n_facts=2)
    assert s["recall"] == 1.0


def test_zero_claims_gives_no_precision_rather_than_a_perfect_one():
    s = F.score_marking({"covered_fact_numbers": [1], "summary_claims": [],
                         "unsupported_claim_numbers": []}, n_facts=2)
    assert s["precision"] is None
    assert s["recall"] == 0.5


def test_missing_keys_do_not_raise():
    s = F.score_marking({}, n_facts=3)
    assert s["recall"] == 0.0 and s["precision"] is None


def test_f1_is_none_when_either_side_is_missing():
    assert F.f1(None, 0.5) is None and F.f1(0.5, None) is None and F.f1(0.0, 0.0) is None
    assert F.f1(0.5, 0.5) == pytest.approx(0.5)


# --- aggregation ------------------------------------------------------------

def _item(p, r, n_facts=4, n_claims=4):
    return {"n_facts": n_facts, "n_covered": 0, "n_claims": n_claims,
            "n_unsupported": 0, "precision": p, "recall": r, "query_id": "q"}


def test_aggregate_averages_over_queries():
    a = F.aggregate("sys", "gpt-5.6-sol", "gpt", [_item(1.0, 0.5), _item(0.5, 1.0)], 0)
    assert a["fact_precision"] == pytest.approx(0.75)
    assert a["fact_recall"] == pytest.approx(0.75)
    # mean of per-query F1: both queries are F1(1.0,0.5)=F1(0.5,1.0)=0.6667
    assert a["fact_f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert a["fact_f1_of_means"] == pytest.approx(0.75)


def test_headline_f1_matches_the_rouge_estimator_not_f1_of_means():
    """eval/scorer.py averages PER-QUERY f-measure. Comparing a fact F1 built from
    mean-P and mean-R against a ROUGE built the other way is not like-for-like, and
    it inflated our reported gap to the frontier from ~4% to ~160%."""
    items = [_item(1.0, 0.1), _item(0.1, 1.0)]
    a = F.aggregate("sys", "gpt-5.6-sol", "gpt", items, 0)
    expected = sum(F.f1(i["precision"], i["recall"]) for i in items) / 2
    assert a["fact_f1"] == pytest.approx(expected, abs=1e-4)
    assert a["fact_f1"] != pytest.approx(a["fact_f1_of_means"], abs=1e-3)


def test_total_failure_queries_score_zero_and_are_not_dropped():
    """Reusing f1() per query returned None for a 0/0 query, so total failures were
    EXCLUDED from the mean. That averages over only the queries a system managed to
    score on and flatters the systems that fail most, inflating ours 0.145 -> 0.262."""
    assert F.per_query_f1(0.0, 0.0) == 0.0
    assert F.f1(0.0, 0.0) is None, "the aggregate-combining helper keeps None"
    spiky = [_item(0.0, 0.0)] * 5 + [_item(0.8, 0.8)] * 5
    a = F.aggregate("spiky", "gpt-5.6-sol", "gpt", spiky, 0)
    assert a["fact_f1"] == pytest.approx(0.4, abs=1e-4), "half the queries scored 0"


def test_a_summary_with_no_checkable_claims_scores_zero_not_none():
    """Precision is undefined with zero claims, but the summary still conveyed
    nothing, so the query is a 0 rather than a query that did not happen."""
    assert F.per_query_f1(None, 0.0) == 0.0
    assert F.per_query_f1(None, 0.5) == 0.0


def test_unmeasurable_recall_stays_none():
    assert F.per_query_f1(0.5, None) is None


def test_aggregate_skips_none_sided_items_without_crashing():
    a = F.aggregate("sys", "gpt-5.6-sol", "gpt", [_item(None, 0.5), _item(0.5, 0.5)], 0)
    assert a["fact_precision"] == pytest.approx(0.5)
    assert a["fact_recall"] == pytest.approx(0.5)


def test_aggregate_reports_unparsed_separately_from_scored():
    a = F.aggregate("sys", "gpt-5.6-sol", "gpt", [_item(1.0, 1.0)], unparsed=3)
    assert a["n_scored"] == 1 and a["unparsed"] == 3


def test_self_preference_flagged_only_when_judge_marks_its_own_vendor():
    assert F.aggregate("gpt-5.6-luna", "gpt-5.6-sol", "gpt", [], 0)["self_preference_risk"]
    assert not F.aggregate("claude-opus-5", "gpt-5.6-sol", "gpt", [], 0)["self_preference_risk"]
    assert not F.aggregate(P.OURS, "gpt-5.6-sol", "gpt", [], 0)["self_preference_risk"]
    assert not F.aggregate(P.OURS, "claude-opus-5", "claude", [], 0)["self_preference_risk"]


# --- wiring -----------------------------------------------------------------

def test_every_panel_system_plus_ours_is_scored():
    assert set(F.systems_map()) == {P.OURS, *P.PANEL}
    assert all(p.exists() for p in F.systems_map().values())


def test_gold_reference_is_available_as_the_calibration_row():
    m = F.systems_map(include_gold=True)
    assert set(m) == {P.OURS, *P.PANEL, F.GOLD}
    assert m[F.GOLD].exists()


def test_gold_row_is_opt_in_so_it_never_lands_in_the_system_table_by_accident():
    assert F.GOLD not in F.systems_map()


def test_community_checkpoints_are_available_and_opt_in():
    """The cheap generality test: do OTHER fine-tuned QMSum systems show the same
    ROUGE-vs-fact divergence, or is it ours alone? Predictions already exist."""
    assert not set(F.COMMUNITY) & set(F.systems_map())
    m = F.systems_map(include_community=True)
    assert set(F.COMMUNITY) <= set(m)
    for s in F.COMMUNITY:
        assert m[s].exists(), f"missing community preds for {s}"


def test_community_defaults_stay_out_of_the_default_run():
    """A default `run` must keep scoring the same six systems, or the headline
    table silently changes shape between sessions."""
    default = [s for s in F.systems_map(include_gold=True, include_community=True)
               if s not in (F.GOLD, *F.COMMUNITY)]
    assert set(default) == {P.OURS, *P.PANEL}


def test_gold_reference_marking_recovers_its_own_facts():
    """Marking the reference against facts extracted from it is a tautology the
    scorer must reproduce. If this drops, the marking step is lossy and every
    row has to be read against the measured ceiling, not 1.0."""
    s = F.score_marking({"covered_fact_numbers": [1, 2, 3],
                         "summary_claims": ["a", "b", "c"],
                         "unsupported_claim_numbers": []}, n_facts=3)
    assert s["recall"] == 1.0 and s["precision"] == 1.0


def test_full_split_run_does_not_overwrite_the_default_sample_row():
    """A 281-query run once clobbered the n=60 row and put both in one table.
    Sample size must be part of the key, not just a column in the output."""
    from eval import protocol
    default = F.result_tag("sys", "gpt-5.6-sol", protocol.JUDGE_SAMPLE_N)
    full = F.result_tag("sys", "gpt-5.6-sol", 281)
    assert default != full and full.endswith("__n281")
    assert "__n" not in default, "the default n must keep its unsuffixed filename"


def test_mixed_sample_sizes_are_flagged_loudly_in_the_rendered_table():
    runs = [F.aggregate("a", "gpt-5.6-sol", "gpt", [_item(0.5, 0.5)] * 60, 0),
            F.aggregate("b", "gpt-5.6-sol", "gpt", [_item(0.5, 0.5)] * 281, 0)]
    md = F.render_markdown(runs)
    assert "WARNING" in md and "NOT comparable" in md


def test_uniform_sample_sizes_produce_no_warning():
    runs = [F.aggregate("a", "gpt-5.6-sol", "gpt", [_item(0.5, 0.5)] * 60, 0),
            F.aggregate("b", "gpt-5.6-sol", "gpt", [_item(0.5, 0.5)] * 60, 0)]
    assert "WARNING" not in F.render_markdown(runs)


def test_markdown_states_the_two_properties_that_justify_this_arm():
    runs = [F.aggregate(P.OURS, "gpt-5.6-sol", "gpt", [_item(0.8, 0.6)], 0)]
    md = F.render_markdown(runs)
    assert "paraphrase" in md and "pointwise" in md
    assert P.OURS in md


# --- split keying, added 2026-07-29 before it could clobber a paid artifact ---

def test_test_split_keeps_its_bare_filenames():
    """Every existing artifact on disk is a test run under the bare name."""
    from eval import judge_facts as F
    assert F.result_tag("sys", "judge", 60) == "sys__judge-judge"
    assert F.result_tag("sys", "judge", 281) == "sys__judge-judge__n281"
    assert F.facts_tag("judge") == "facts__judge"


def test_a_val_run_cannot_overwrite_the_test_row_of_the_same_size():
    from eval import judge_facts as F
    assert F.result_tag("sys", "judge", 60, "val") != F.result_tag("sys", "judge", 60, "test")
    assert F.result_tag("sys", "judge", 60, "val").endswith("__val")


def test_a_val_extraction_cannot_overwrite_the_paid_test_fact_decomposition():
    """facts__gpt-5.6-sol.json holds 2,067 facts over 281 test references and cost
    real money. An unsuffixed val extraction would have replaced it with 272."""
    from eval import judge_facts as F
    assert F.facts_tag("judge", "val") == "facts__judge__val"
    assert F.facts_tag("judge", "val") != F.facts_tag("judge", "test")


# --- the incoherent-marking defect, root-caused 2026-07-29 -------------------

def test_a_marking_that_covers_facts_but_supports_no_claim_is_flagged():
    """The judge says the summary conveys facts 1 and 3, and in the same reply
    says both its claims are unsupported by those facts. Real case: test_0_s8."""
    from eval import judge_facts as F
    r = F.score_marking({"covered_fact_numbers": [1, 3],
                         "summary_claims": ["a", "b"],
                         "unsupported_claim_numbers": [1, 2]}, n_facts=4)
    assert r["incoherent"] is True
    assert r["recall"] == 0.5 and r["precision"] == 0.0


def test_a_coherent_marking_is_not_flagged():
    from eval import judge_facts as F
    r = F.score_marking({"covered_fact_numbers": [1],
                         "summary_claims": ["a", "b"],
                         "unsupported_claim_numbers": [2]}, n_facts=4)
    assert r["incoherent"] is False


def test_covering_nothing_and_supporting_nothing_is_coherent():
    """A summary that misses everything is a legitimate zero, not a defect."""
    from eval import judge_facts as F
    r = F.score_marking({"covered_fact_numbers": [],
                         "summary_claims": ["a"],
                         "unsupported_claim_numbers": [1]}, n_facts=4)
    assert r["incoherent"] is False


def test_a_summary_with_no_claims_is_not_flagged_as_incoherent():
    """precision is None there, which is a different problem with its own test."""
    from eval import judge_facts as F
    r = F.score_marking({"covered_fact_numbers": [1], "summary_claims": [],
                         "unsupported_claim_numbers": []}, n_facts=4)
    assert r["incoherent"] is False
    assert r["precision"] is None


def test_aggregate_reports_the_incoherence_rate_beside_precision():
    """Precision must not be quotable without its reliability caveat attached."""
    from eval import judge_facts as F
    items = [F.score_marking({"covered_fact_numbers": [1], "summary_claims": ["a"],
                              "unsupported_claim_numbers": [1]}, 4),
             F.score_marking({"covered_fact_numbers": [1], "summary_claims": ["a"],
                              "unsupported_claim_numbers": []}, 4)]
    for it, q in zip(items, ["q1", "q2"]):
        it["query_id"] = q
    a = F.aggregate("sys", "judge", "gpt", items, 0)
    assert a["incoherent_markings"] == 1
    assert a["incoherent_rate"] == 0.5
    assert a["mean_claims_denominator"] == 1.0
    assert "denominator" in a["precision_caveat"]


def test_aggregate_tolerates_stored_items_predating_the_flag():
    """17 result files on disk were written before `incoherent` existed."""
    from eval import judge_facts as F
    old = [{"n_facts": 4, "n_covered": 1, "n_claims": 2, "n_unsupported": 1,
            "recall": 0.25, "precision": 0.5, "query_id": "q"}]
    a = F.aggregate("sys", "judge", "gpt", old, 0)
    assert a["incoherent_markings"] == 0
    assert a["fact_precision"] == 0.5
