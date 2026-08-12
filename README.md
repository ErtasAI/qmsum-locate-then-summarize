# QMSum locate-then-summarize

Code, frozen protocol, per-query predictions and released models for the paper
*Locate-then-summarize on QMSum: training regime and architecture outweigh scale in
query-focused multi-domain meeting summarization* (Ertas AI, in preparation).

The system is a two-stage pipeline for query-focused meeting summarization on
[QMSum](https://github.com/Yale-LILY/QMSum): a cross-encoder **locator** scores fixed-width
windows of the transcript against the query and packs the best ones to a word budget, then a
LoRA fine-tuned **summarizer** (LFM2.5 1.2B) answers the query from those spans alone. Every
number in the paper traces to a row in `results/rows/` and a per-query prediction file in
`results/preds/`, all generated under the frozen protocol in `eval/protocol.py` and scored by
one frozen scorer.

## Released models

| Artifact | Hub repo | Pinned revision |
|---|---|---|
| Summarizer LoRA adapter | [ErtasAI/qmsum-summarizer-lfm2.5-1.2b-lora](https://huggingface.co/ErtasAI/qmsum-summarizer-lfm2.5-1.2b-lora) | `7d73ab65` |
| Locator, promoted (L12, 375-word windows) | [ErtasAI/qmsum-locator-minilm-l12-w375](https://huggingface.co/ErtasAI/qmsum-locator-minilm-l12-w375) | `f54ed53f` |
| Locator, protocol-exact (L6, 900-word windows) | [ErtasAI/qmsum-locator-minilm-l6-w900](https://huggingface.co/ErtasAI/qmsum-locator-minilm-l6-w900) | `a1b9e64d` |

`scripts/fetch_artifacts.py` downloads all three at exactly these revisions and verifies each
weight file's SHA256, so what you run is what the paper measured.

## Replicate the headline rows

Requirements: Python 3.12, an NVIDIA GPU (peak inference VRAM is 5.73 GB; any 8 GB card is
comfortable), about 4 GB of downloads (base model, artifacts, data). No API keys.

```bash
git clone https://github.com/ErtasAI/qmsum-locate-then-summarize
cd qmsum-locate-then-summarize
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 1. Data: fetch QMSum from the official repo and build the normalized splits
python data/download_qmsum.py
python -m data.normalize

# 2. Models: fetch the released artifacts at pinned revisions, hash-verified
python scripts/fetch_artifacts.py

# 3. Run the promoted configuration on the official test split (~15 min on an RTX 5070 Ti)
python -m pipeline.run_pipeline --adapter checkpoints/m3-fusion-lfm2.5-1.2b/final \
    --split test --locator crossencoder \
    --locator-ckpt checkpoints/locator-crossencoder-w375-l12 \
    --chunk-words 375 --span-budget 2000 \
    --out results/preds/repro-promoted-test.jsonl

# 4. Score it (first --bertscore run downloads roberta-large, ~1.4 GB)
python -m eval.run_eval --pred results/preds/repro-promoted-test.jsonl \
    --split test --run-id repro-promoted-test --bertscore
```

For the protocol-exact row, the pipeline's defaults ARE the frozen protocol, so step 3 needs
no locator arguments:

```bash
python -m pipeline.run_pipeline --adapter checkpoints/m3-fusion-lfm2.5-1.2b/final \
    --split test --out results/preds/repro-protocol-test.jsonl
python -m eval.run_eval --pred results/preds/repro-protocol-test.jsonl \
    --split test --run-id repro-protocol-test --bertscore
```

For a two-minute path check before committing to the full split, add `--limit 5` to the
pipeline command and `--allow-partial` to the scoring command.

### Expected results

Full official test split, n=281, greedy decoding:

| Configuration | Published row | R1 | R2 | R-L | R-Lsum | BERTScore |
|---|---|---|---|---|---|---|
| Promoted | `results/rows/loc-w375-l12-b2000-test.json` | 0.3541 | 0.1228 | 0.2463 | 0.3136 | 0.8733 |
| Protocol-exact | `results/rows/m3-fusion-lfm2.5-1.2b-test.json` | 0.3339 | 0.1065 | 0.2283 | 0.2930 | 0.8680 |

This exact path was verified on 2026-08-12: a fresh extraction of this repo, the data build,
the hash-verified artifact fetch and the promoted-configuration run reproduced all 281
predictions byte-identically against the published prediction file and matched the published
row to full float precision (on the GPU family that produced the rows). On different
hardware, small numeric drift in the last decimal places is possible; the ROUGE figures
should match to about three decimal places. Wall-clock
latency reproduces only to about 15% even on identical hardware, so treat any latency
comparison under 20% as noise.

`results/CHART.md` holds every row the project scored, including five frontier zero-shot
baselines and four community checkpoints under the same frozen scorer, plus a separate,
deliberately incommensurable table of published numbers from the literature.

## Optional legs (API keys)

The frontier baselines and the fact-level judge are re-runnable but need keys. Copy
`.env.example` to `.env` and fill in what you need; each key's role is documented in that
file. Cached baseline predictions for all five frontier rows are already in `results/preds/`,
so you can rescore them without any key:

```bash
python -m eval.run_eval --pred results/preds/baseline-gpt-5.6-luna-test.jsonl \
    --split test --run-id rescore-luna --bertscore
```

## Repo map

| Path | What it is |
|---|---|
| `eval/protocol.py` | The frozen protocol: seed, budgets, prompt template. Locked 2026-07-23; `tests/test_protocol_frozen.py` enforces it |
| `eval/scorer.py`, `eval/run_eval.py` | The frozen scorer (rouge-score 0.1.2, bert-score 0.3.13) and the row writer |
| `eval/paired_bootstrap.py` | Paired bootstrap CIs for any two prediction files |
| `pipeline/` | Chunker, both locators, and the end-to-end pipeline runner |
| `training/train_lora.py` + `training-configs/` | QLoRA training with full instrumentation; one YAML per trained configuration |
| `data/` | QMSum download, normalization, and training-data builders |
| `results/rows/` | One JSON per scored run, the paper's tables' source of truth |
| `results/preds/` | Per-query predictions for every row, including all baselines |
| `external/query-focused-sum/` | Ported model classes for the Socratic SegEnc comparison rows (BSD-3, Salesforce; licence retained) |
| `tests/` | The test suite; `pytest` from the repo root |

## Retraining caveats, stated up front

The released adapter is the artifact the paper measures; retraining does not reproduce it.
Training-seed dispersion on this benchmark is 1.59 ROUGE-1, and on the 16 GB card the paper
used, training runs in host-memory fallback: of four attempted seeds, one completed. Training
takes about five and a half GPU-hours when it completes. The locators are cheap to retrain
(about 7 and 1.5 minutes) but a retrain is likewise a different checkpoint. Use
`scripts/fetch_artifacts.py` for replication; retrain only to study the training itself.

## Tests

```bash
python -m pytest tests/
```

The suite runs without a GPU and without API keys, but expects the data step above to have
run first (it exercises the builders and the built splits). From a fresh clone plus the data
step: 351 passed, 2 skipped.

## Licences

- **Code:** Apache 2.0 (`LICENSE`).
- **`external/query-focused-sum/`:** BSD 3-Clause, Salesforce (`external/query-focused-sum/LICENSE.txt`), ported from
  [salesforce/query-focused-sum](https://github.com/salesforce/query-focused-sum).
- **Models:** each released artifact carries its licence on its Hub page. The locators are
  Apache 2.0; the summarizer adapter inherits the LFM Open License v1.0 from its base model,
  which caps free commercial use at US$10M annual revenue.
- **Data:** QMSum is MIT; it draws on the AMI and ICSI meeting corpora (CC BY 4.0). This repo
  redistributes no transcript text; `data/download_qmsum.py` fetches it from the official
  QMSum repo.

## Citation

The paper is in preparation; a citation entry will land here with the preprint. Until then,
if you use the pipeline or the released models, please cite QMSum as well:

```bibtex
@inproceedings{zhong2021qmsum,
  title     = {{QMSum}: A New Benchmark for Query-based Multi-domain Meeting Summarization},
  author    = {Zhong, Ming and Yin, Da and Yu, Tao and Zaidi, Ahmad and Mutuma, Mutethia and
               Jha, Rahul and Awadallah, Ahmed Hassan and Celikyilmaz, Asli and Liu, Yang and
               Qiu, Xipeng and Radev, Dragomir},
  booktitle = {Proceedings of NAACL-HLT},
  year      = {2021}
}
```

---

Released by [Ertas AI](https://www.ertas.ai). We build custom small models that run on-device
and in your own infrastructure.
