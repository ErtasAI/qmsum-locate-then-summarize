"""Pins the optional stock and span-trained SegEnc comparison artifacts."""
from scripts.fetch_artifacts import COMPARISON_ARTIFACTS, PIPELINE_ARTIFACTS, artifacts_for


def test_default_fetch_remains_the_three_headline_pipeline_artifacts():
    assert artifacts_for(include_comparisons=False) == PIPELINE_ARTIFACTS
    assert len(PIPELINE_ARTIFACTS) == 3


def test_comparison_fetch_pins_both_segenc_checkpoints_and_weight_hashes():
    by_repo = {artifact["repo"]: artifact for artifact in COMPARISON_ARTIFACTS}
    span = by_repo["ErtasAI/qmsum-summarizer-segenc-406m-spans"]
    assert span["revision"] == "26abdfc1b0373c21b54dfd5584eb432e9e9d0a40"
    assert span["dest"] == "checkpoints/segenc-spans-c8/epoch8"
    assert span["sha256"]["model.safetensors"] == (
        "84ac9a6b86d6290fd37ff86ba5f0e3a10adadf8e89f265929817e524cd758013")

    stock = by_repo["Salesforce/socratic-pretraining-qmsum"]
    assert stock["revision"] == "d127cbc54b974a58e8bd75935f2863d136e0bc3d"
    assert stock["dest"] == "results/external/socratic-qmsum"
    assert stock["sha256"]["pytorch_model.bin"] == (
        "5de3c9e7c4d9a274b423c44d018b8cc768dfd5ada7029c12af82076a20c3876d")


def test_include_comparisons_adds_without_replacing_pipeline_artifacts():
    assert artifacts_for(include_comparisons=True) == (
        PIPELINE_ARTIFACTS + COMPARISON_ARTIFACTS)
