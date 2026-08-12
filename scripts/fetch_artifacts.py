"""Fetch the released model artifacts from the Hub at pinned revisions, verified by hash.

Downloads the three artifacts every published system row depends on, into the exact
local paths the pipeline's defaults and the paper's run ids reference:

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

No API key needed; the repos are public.
"""
import hashlib
import pathlib
import sys

from huggingface_hub import snapshot_download

ROOT = pathlib.Path(__file__).resolve().parents[1]

ARTIFACTS = [
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
    failures = 0
    for a in ARTIFACTS:
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
