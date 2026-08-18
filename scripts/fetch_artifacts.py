"""Fetch the released model artifacts from the Hub at pinned revisions, verified by hash.

By default, downloads the three artifacts every published pipeline row depends on. Pass
``--include-comparisons`` to also fetch the stock and span-trained SegEnc checkpoints used by
the apples-to-apples comparison figure.

    Hub repo                                    -> local path
    ErtasAI/qmsum-summarizer-lfm2.5-1.2b-lora   -> checkpoints/m3-fusion-lfm2.5-1.2b/final
    ErtasAI/qmsum-locator-minilm-l6-w900        -> checkpoints/locator-crossencoder
    ErtasAI/qmsum-locator-minilm-l12-w375       -> checkpoints/locator-crossencoder-w375-l12

The local names are the training-time checkpoint names, kept so that every result row's
run id, every prediction file's provenance fields, and the pipeline's default paths keep
pointing at something that exists. Each download is pinned to the exact repo revision
published for the paper, and the weight file's SHA256 is verified after download, so a
later force-push to a repo cannot silently change what this script installs.

Usage:
    python scripts/fetch_artifacts.py
    python scripts/fetch_artifacts.py --include-comparisons

No API key needed; the repos are public.
"""
import argparse
import hashlib
import pathlib
import sys

from huggingface_hub import snapshot_download

ROOT = pathlib.Path(__file__).resolve().parents[1]

PIPELINE_ARTIFACTS = [
    {
        "repo": "ErtasAI/qmsum-summarizer-lfm2.5-1.2b-lora",
        "revision": "7d73ab6535ea4e33c352f46105649c195fa50543",
        "dest": "checkpoints/m3-fusion-lfm2.5-1.2b/final",
        "sha256": {
            "adapter_model.safetensors":
                "75f871fa845ad90d02b5e2986269dd646faef1042000c796923b4d435e8759ed",
        },
    },
    {
        "repo": "ErtasAI/qmsum-locator-minilm-l6-w900",
        "revision": "a1b9e64d82cadb1199ed36d991e06efbb9e5dcd3",
        "dest": "checkpoints/locator-crossencoder",
        "sha256": {
            "model.safetensors":
                "6588d79e1db83bc0b8aa9901311371927e4e2bfe155d2965d965c6815acccfad",
        },
    },
    {
        "repo": "ErtasAI/qmsum-locator-minilm-l12-w375",
        "revision": "f54ed53f816b14bf6b129cee25f10be08bc16bdd",
        "dest": "checkpoints/locator-crossencoder-w375-l12",
        "sha256": {
            "model.safetensors":
                "5adbc3238603476ff9516101fb571fb8526bf67a22ff2a716985f20205cb053d",
        },
    },
]

COMPARISON_ARTIFACTS = [
    {
        "repo": "ErtasAI/qmsum-summarizer-segenc-406m-spans",
        "revision": "26abdfc1b0373c21b54dfd5584eb432e9e9d0a40",
        "dest": "checkpoints/segenc-spans-c8/epoch8",
        "sha256": {
            "model.safetensors":
                "84ac9a6b86d6290fd37ff86ba5f0e3a10adadf8e89f265929817e524cd758013",
        },
    },
    {
        "repo": "Salesforce/socratic-pretraining-qmsum",
        "revision": "d127cbc54b974a58e8bd75935f2863d136e0bc3d",
        "dest": "results/external/socratic-qmsum",
        "sha256": {
            "pytorch_model.bin":
                "5de3c9e7c4d9a274b423c44d018b8cc768dfd5ada7029c12af82076a20c3876d",
        },
    },
]


def artifacts_for(include_comparisons=False):
    return PIPELINE_ARTIFACTS + (COMPARISON_ARTIFACTS if include_comparisons else [])


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-comparisons", action="store_true",
                        help="also fetch stock and span-trained SegEnc checkpoints (~3.2 GB)")
    args = parser.parse_args()
    failures = 0
    for a in artifacts_for(args.include_comparisons):
        dest = ROOT / a["dest"]
        print(f"fetching {a['repo']} @ {a['revision'][:8]} -> {a['dest']}")
        snapshot_download(a["repo"], revision=a["revision"], local_dir=dest)
        for fname, want in a["sha256"].items():
            got = file_sha256(dest / fname)
            if got == want:
                print(f"  sha256 OK  {fname}")
            else:
                failures += 1
                print(f"  sha256 MISMATCH  {fname}\n    want {want}\n    got  {got}")
    if failures:
        sys.exit(f"{failures} hash check(s) failed; do not use the downloaded artifacts")
    print("all artifacts fetched and verified")


if __name__ == "__main__":
    main()
