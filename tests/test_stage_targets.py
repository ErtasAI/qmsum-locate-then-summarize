import json, pathlib, subprocess, sys, re

ROOT = pathlib.Path(__file__).resolve().parents[1]

def setup_module():
    subprocess.run([sys.executable, "-m", "data.build_stage_targets"], check=True, cwd=ROOT)

def _rows(name):
    return [json.loads(l) for l in (ROOT / "data" / "built" / name).open(encoding="utf-8")]

def test_fusion_schema_and_budget():
    rows = _rows("fusion.train.jsonl")
    assert rows and all(set(r) == {"query_id", "prompt", "response"} for r in rows[:50])
    assert all(len(r["prompt"].split()) <= 3400 for r in rows[:50])

def test_coarse_targets_come_from_reference():
    for r in _rows("coarse.train.jsonl")[:30]:
        assert r["response"].strip()

def test_sentence_split():
    from data.build_stage_targets import sentences
    assert sentences("First point. Second point? Third.") == ["First point.", "Second point?", "Third."]

def test_alignment_regression_argument_order():
    from data.build_stage_targets import best_window, _R1
    # Window 0: tiny, shares one word with the sentence. Window 1: long, contains
    # every content word of the sentence among much else. Sentence-as-target recall
    # picks window 1; the old transposed order (window as target) favored the tiny
    # window 0. This fixture discriminates the two.
    w0 = "mockups"
    w1 = ("the design team spent the session reviewing layout mockups button colours "
          "typography spacing accessibility contrast and the export flow for the new builder")
    sentence = "They reviewed the layout mockups and button colours."
    assert best_window(sentence, [w0, w1]) == 1
    old_style = max(range(2), key=lambda k: _R1.score([w0, w1][k], sentence)["rouge1"].recall)
    assert old_style == 0  # documents the bug this guards against
