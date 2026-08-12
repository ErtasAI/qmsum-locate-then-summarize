import json, pathlib

# test = 281 in the RELEASED official data (Yale-LILY repo; corroborated by the
# HF mirror and follow-up papers, e.g. QontSum). The paper's table says 279;
# the released split is our protocol substrate. Reconciled 2026-07-23.
RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = {"train": (162, 1257), "val": (35, 272), "test": (35, 281)}

def _counts(split):
    rows = [json.loads(l) for l in (RAW / f"{split}.jsonl").open(encoding="utf-8")]
    pairs = sum(len(r["general_query_list"]) + len(r["specific_query_list"]) for r in rows)
    return len(rows), pairs

def test_split_counts():
    for split, expected in EXPECTED.items():
        assert _counts(split) == expected, f"{split}: got {_counts(split)}"
