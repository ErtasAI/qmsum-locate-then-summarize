"""Fetch the official QMSum split files from Yale-LILY/QMSum."""
import pathlib, requests

BASE = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl"
RAW = pathlib.Path(__file__).resolve().parent / "raw"

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        dest = RAW / f"{split}.jsonl"
        if dest.exists():
            print(f"{split}: already present, skipping")
            continue
        r = requests.get(f"{BASE}/{split}.jsonl", timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"{split}: {len(r.content)} bytes")

if __name__ == "__main__":
    main()
