from eval.scorer import score

REFS = [
    "The team agreed to move the launch to May and assigned QA to Dana.",
    "Marketing will draft the landing page copy before Friday's review.",
    "The budget discussion was postponed until the next quarterly meeting.",
]

def test_identity_scores_near_perfect():
    s = score(REFS, REFS)
    assert s["rouge1"] > 0.99 and s["rougeL"] > 0.99

def test_shuffled_scores_low():
    shuffled = [REFS[1], REFS[2], REFS[0]]
    s = score(shuffled, REFS)
    assert s["rouge1"] < 0.35

def test_keys_present():
    s = score(REFS, REFS)
    assert set(s) >= {"rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore_f1"}
    assert s["bertscore_f1"] is None
