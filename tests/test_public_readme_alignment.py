"""Prevents the public narrative from drifting behind the reviewed paper."""
import pathlib


README = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
NORMALIZED = " ".join(README.split())


def test_title_and_segenc_boundary_match_the_reviewed_paper():
    assert "Retrieved-Span Training for Efficient Query-Focused Meeting Summarization on QMSum" in README
    assert "git clone https://github.com/ErtasAI/qmsum-retrieved-span-training" in README
    assert "cd qmsum-retrieved-span-training" in README
    assert "35.30" in README and "38.60" in README
    assert "preprocessing, decoding, implementation, or checkpoint differences" in NORMALIZED


def test_comparison_table_uses_compact_names_and_includes_the_port_row():
    table_start = README.index("| System | Params | Reads |")
    table_end = README.index("\n\n¹ ", table_start)
    table_lines = [
        line for line in README[table_start:table_end].splitlines() if line.startswith("|")
    ]
    system_names = [
        line.split("|", 2)[1].strip().replace("**", "").rstrip("¹²³⁴")
        for line in table_lines[2:]
    ]

    assert system_names == [
        "Socratic-SegEnc (released outputs)",
        "Span-trained SegEnc",
        "Ours (promoted)",
        "Socratic-SegEnc (our port)",
        "Ours (protocol-exact)",
        "GPT-5.6 Luna (zero-shot)",
        "GPT-5.6 Sol (zero-shot)",
        "LFM2.5-1.2B (zero-shot, spans)",
        "Claude Opus 5 (zero-shot)",
        "Claude Haiku 4.5 (zero-shot)",
        "DistilBART",
        "LFM2.5-1.2B (zero-shot, truncated)",
        "Claude Sonnet 5 (zero-shot)",
        "BART-large-CNN",
        "PEGASUS",
        "LED-base",
    ]
    assert (
        "| Socratic-SegEnc (our port) | 406M | transcript up to ~11.8k words | "
        "0.3530 | 0.1187 | 0.3056 | 0.8695 | N/A |"
    ) in README


def test_uncertainty_language_does_not_claim_equivalence_or_a_seed_distribution():
    assert "statistically indistinguishable" not in README
    assert "Training-seed dispersion" not in README
    assert "does not establish equivalence" in NORMALIZED
    assert "two-run observed range" in README
    assert "2.8x fewer parameters" in README


def test_central_meeting_cluster_intervals_and_limitations_are_disclosed():
    assert "meeting-cluster bootstrap" in README
    assert "results/intervals/" in README
    assert "output length was not experimentally controlled" in NORMALIZED.lower()
    assert "human or factuality evaluation" in NORMALIZED


def test_exact_segenc_reproduction_routes_are_documented():
    assert "--include-comparisons" in README
    assert "python -m training.segenc_recipe" in README
    assert "--max-num-chunks 32" in README
    assert "--max-num-chunks 8" in README
