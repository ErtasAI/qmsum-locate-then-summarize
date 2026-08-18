"""Guards the public, dependency-free rendering of the paper's Figure 1 comparison."""
import pathlib
import xml.etree.ElementTree as ET

from results.build_figure1 import BASE_CHECKPOINT, OURS, SCALE, build_svg


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_figure_uses_the_executable_stock_port_not_author_predictions():
    by_run = {item[1]: item for item in SCALE}
    assert by_run["segenc-port-full-test"][3] == BASE_CHECKPOINT
    assert by_run["segenc-c8cont-e4-test"][3] == OURS
    assert "row-socratic-segenc-test" not in by_run


def test_svg_is_valid_deterministic_and_carries_reviewed_labels():
    first = build_svg()
    second = build_svg()
    assert first == second
    ET.fromstring(first)
    assert first.count("Size undisclosed") == 5
    assert "35.30" in first  # stock Socratic SegEnc under our executable port
    assert "36.33" in first  # SegEnc trained on our span regime
    assert "38.60" not in first  # authors' released predictions are not apples-to-apples here
    assert "stock Socratic SegEnc" in first
    assert "span-trained SegEnc" in first


def test_readme_embeds_the_generated_figure():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "results/figure1-scale.svg" in readme
