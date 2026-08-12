from pipeline.chunker import windows

UTTS = [{"speaker": "A", "text": "alpha " * 100}] * 30  # 100 words each

def test_windows_cover_everything_in_order():
    w = windows(UTTS)
    assert w[0]["start"] == 0 and w[-1]["end"] == 29
    for a, b in zip(w, w[1:]):
        assert b["start"] == a["end"] + 1

def test_window_budget_respected():
    for w in windows(UTTS):
        assert len(w["text"].split()) <= 900
