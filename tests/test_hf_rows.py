from eval.hf_rows import CHECKPOINTS, build_input

def test_checkpoint_registry():
    assert len(CHECKPOINTS) == 4
    assert all("/" in c["repo"] and c["slug"] for c in CHECKPOINTS)

def test_build_input_contains_query_and_respects_budget():
    utts = [{"speaker": "A", "text": "word " * 30}] * 200
    text = build_input("What was decided?", utts, word_budget=800)
    assert "What was decided?" in text and len(text.split()) <= 830
