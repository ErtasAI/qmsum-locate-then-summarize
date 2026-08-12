from pipeline.chunker import windows
from pipeline.locator_embed import select

def test_select_relevant_window_and_budget():
    utts = ([{"speaker": "A", "text": "filler chatter about lunch plans " * 20}] * 8
            + [{"speaker": "B", "text": "the launch date decision was moved to May for QA reasons " * 20}]
            + [{"speaker": "A", "text": "more filler about parking " * 20}] * 8)
    w = windows(utts)
    picked = select("When is the launch date?", w, budget_words=1500)
    assert any("launch date decision" in p["text"] for p in picked)
    assert sum(len(p["text"].split()) for p in picked) <= 1500
    starts = [p["start"] for p in picked]
    assert starts == sorted(starts)
