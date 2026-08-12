"""Panel arithmetic. Every number here ends up in a published table, so the
summary maths is tested independently of any API call."""
import pytest
from eval import run_judge_panel as P


def _run(contestant, judge_model, a_wins, b_wins, ties, unparsed=0, items=None,
         self_pref=False, cost=1.0):
    n_decided = a_wins + b_wins + ties
    return {"contestant": contestant, "ours": P.OURS, "split": "test",
            "judge_model": judge_model, "judge_vendor": P.J.vendor_of(judge_model),
            "a_wins": a_wins, "b_wins": b_wins, "ties": ties, "unparsed": unparsed,
            "truncated": 0, "n_decided": n_decided, "n_sampled": n_decided + unparsed,
            "self_preference_risk": self_pref, "cost_usd": cost,
            "entries_without_usage": 0, "items": items or []}


def _items(winners):
    return [{"query_id": f"q{i}", "swap": i % 2 == 1, "verdict": "X",
             "winner": w, "truncated": False} for i, w in enumerate(winners)]


# --- panel wiring -----------------------------------------------------------

def test_panel_covers_every_frontier_baseline_run_in_m0():
    assert set(P.PANEL) == {"gpt-5.6-luna", "gpt-5.6-sol", "claude-opus-5",
                            "claude-sonnet-5", "claude-haiku-4-5"}

def test_every_pair_has_exactly_one_clean_cross_vendor_judge():
    """The whole design rests on this: with a GPT judge and a Claude judge, each
    frontier contestant is scored by exactly one judge from another vendor."""
    for model in P.PANEL:
        contestant_vendor = P.J.vendor_of(model)
        judges = [P.J.vendor_of(P.J.DEFAULT_JUDGE_MODEL[p]) for p in P.JUDGE_PROVIDERS]
        clean = [v for v in judges if v != contestant_vendor]
        assert len(clean) == 1, f"{model} has {len(clean)} clean judges, expected 1"

def test_default_judges_are_one_per_vendor():
    vendors = [P.J.vendor_of(P.J.DEFAULT_JUDGE_MODEL[p]) for p in P.JUDGE_PROVIDERS]
    assert sorted(vendors) == ["anthropic", "openai"]

def test_pred_paths_point_at_real_files():
    assert P.ours_preds().exists()
    for m in P.PANEL:
        assert P.theirs_preds(m).exists(), f"missing baseline preds for {m}"


# --- win rates --------------------------------------------------------------

def test_win_rate_excludes_unparsed_from_the_denominator():
    r = _run("claude-opus-5", "gpt-5.6-sol", a_wins=30, b_wins=20, ties=10, unparsed=5)
    assert P.win_rate(r, "a") == pytest.approx(30 / 60)

def test_win_rate_is_none_when_nothing_was_decided():
    assert P.win_rate(_run("claude-opus-5", "gpt-5.6-sol", 0, 0, 0, unparsed=4), "a") is None


# --- agreement --------------------------------------------------------------

def test_agreement_is_perfect_when_judges_match():
    a = _run("claude-opus-5", "gpt-5.6-sol", 2, 1, 1, items=_items(["a", "b", "a", "tie"]))
    b = _run("claude-opus-5", "claude-opus-5", 2, 1, 1, items=_items(["a", "b", "a", "tie"]))
    ag = P.agreement(a, b)
    assert ag["raw"] == 1.0 and ag["kappa"] == 1.0 and ag["n_comparable"] == 4

def test_agreement_ignores_items_either_judge_failed_to_parse():
    a = _run("claude-opus-5", "gpt-5.6-sol", 2, 0, 0, items=_items(["a", "a", None, "b"]))
    b = _run("claude-opus-5", "claude-opus-5", 2, 0, 0, items=_items(["a", "a", "b", None]))
    assert P.agreement(a, b)["n_comparable"] == 2

def test_kappa_is_zero_when_agreement_is_only_what_chance_predicts():
    both = [("a", "a")] * 25 + [("a", "b")] * 25 + [("b", "a")] * 25 + [("b", "b")] * 25
    assert P.cohens_kappa(both) == pytest.approx(0.0, abs=1e-9)


# --- self-preference --------------------------------------------------------

def test_self_preference_delta_is_positive_when_a_vendor_favours_its_own():
    """GPT contestant: the Claude judge is clean, the GPT judge is not. If the
    GPT judge rates the GPT contestant higher, the delta must come out positive."""
    clean = _run("gpt-5.6-sol", "claude-opus-5", a_wins=40, b_wins=20, ties=0)
    dirty = _run("gpt-5.6-sol", "gpt-5.6-sol", a_wins=25, b_wins=35, ties=0, self_pref=True)
    row = P.summarize([clean, dirty])["pairs"][0]
    assert row["clean_judge"] == "claude-opus-5"
    assert row["same_vendor_judge"] == "gpt-5.6-sol"
    assert row["contestant_win_rate_clean_judge"] == pytest.approx(20 / 60, abs=1e-4)
    assert row["contestant_win_rate_same_vendor_judge"] == pytest.approx(35 / 60, abs=1e-4)
    assert row["self_preference_delta"] == pytest.approx(0.25)

def test_reported_row_comes_from_the_clean_judge_not_the_same_vendor_one():
    clean = _run("claude-opus-5", "gpt-5.6-sol", a_wins=40, b_wins=20, ties=0)
    dirty = _run("claude-opus-5", "claude-opus-5", a_wins=10, b_wins=50, ties=0, self_pref=True)
    row = P.summarize([clean, dirty])["pairs"][0]
    assert row["ours_wins"] == 40 and row["theirs_wins"] == 20
    assert row["ours_win_rate"] == pytest.approx(40 / 60, abs=1e-4)

def test_summary_counts_pairs_won_on_the_clean_judge_only():
    runs = [_run("gpt-5.6-luna", "claude-opus-5", 40, 20, 0),
            _run("gpt-5.6-luna", "gpt-5.6-sol", 10, 50, 0, self_pref=True),
            _run("claude-opus-5", "gpt-5.6-sol", 20, 40, 0),
            _run("claude-opus-5", "claude-opus-5", 55, 5, 0, self_pref=True)]
    s = P.summarize(runs)
    assert s["pairs_won_on_clean_judge"] == "1/2"

def test_summary_totals_cost_across_every_judge_run():
    runs = [_run("gpt-5.6-luna", "claude-opus-5", 30, 30, 0, cost=1.25),
            _run("gpt-5.6-luna", "gpt-5.6-sol", 30, 30, 0, self_pref=True, cost=0.75)]
    assert P.summarize(runs)["total_cost_usd"] == pytest.approx(2.0)

# --- judging modes ----------------------------------------------------------

def test_preference_mode_keeps_its_original_unsuffixed_tag():
    """Adding the grounded arm must not orphan runs already paid for."""
    assert P.run_tag("claude-opus-5", "gpt-5.6-sol", "preference") == \
           P.run_tag("claude-opus-5", "gpt-5.6-sol")
    assert "preference" not in P.run_tag("claude-opus-5", "gpt-5.6-sol")

def test_modes_write_to_separate_tags():
    a = P.run_tag("claude-opus-5", "gpt-5.6-sol", "preference")
    b = P.run_tag("claude-opus-5", "gpt-5.6-sol", "grounded")
    assert a != b and b.endswith("__grounded")

def test_grounding_lift_is_positive_when_the_reference_helps_us():
    pref = P.summarize([_run("claude-opus-5", "gpt-5.6-sol", 20, 40, 0)])
    grnd = P.summarize([_run("claude-opus-5", "gpt-5.6-sol", 45, 15, 0)])
    c = P.compare_modes(pref, grnd)
    assert c["pairs"][0]["grounding_lift"] == pytest.approx(45 / 60 - 20 / 60, abs=1e-4)
    assert c["mean_grounding_lift"] > 0

def test_mode_comparison_skips_pairs_missing_from_either_mode():
    pref = P.summarize([_run("claude-opus-5", "gpt-5.6-sol", 20, 40, 0),
                        _run("gpt-5.6-luna", "claude-opus-5", 30, 30, 0)])
    grnd = P.summarize([_run("claude-opus-5", "gpt-5.6-sol", 45, 15, 0)])
    assert [r["contestant"] for r in P.compare_modes(pref, grnd)["pairs"]] == ["claude-opus-5"]

def test_markdown_names_the_mode_and_its_caveat():
    s = P.summarize([_run("claude-opus-5", "gpt-5.6-sol", 30, 30, 0)])
    assert "verbosity bias" in P.render_markdown(s)
    g = P.summarize([dict(_run("claude-opus-5", "gpt-5.6-sol", 30, 30, 0), mode="grounded")])
    assert "reference answer" in P.render_markdown(g)

def test_markdown_renders_both_tables():
    runs = [_run("gpt-5.6-luna", "claude-opus-5", 35, 25, 0,
                 items=_items(["a"] * 35 + ["b"] * 25)),
            _run("gpt-5.6-luna", "gpt-5.6-sol", 25, 35, 0, self_pref=True,
                 items=_items(["a"] * 25 + ["b"] * 35))]
    md = P.render_markdown(P.summarize(runs))
    assert "Clean judge" in md and "Self-preference" in md
    assert "gpt-5.6-luna" in md and "claude-opus-5" in md
