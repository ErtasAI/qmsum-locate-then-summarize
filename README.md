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

### The full comparison, one frozen scorer

The paper's headline findings are orderings, and they mean something because every row below
was produced by us and scored by the same frozen scorer on the same official test split
(n=281). Sorted by ROUGE-1:

| System | Params | Reads | R1 | R2 | R-Lsum | BERTScore | API cost, full split² |
|---|---|---|---|---|---|---|---|
| Socratic SegEnc (Pagnoni et al., ACL 2023), authors' released predictions¹ | 406M | transcript up to ~11.8k words | 0.3860 | 0.1391 | 0.3371 | 0.8737 | N/A |
| SegEnc, retrained by us on located spans | 406M | 2,000 located words | 0.3633 | 0.1272 | 0.3217 | 0.8710 | N/A |
| **This system, promoted configuration** | **1.2B + 33M** | **2,000 located words** | **0.3541** | **0.1228** | **0.3136** | **0.8733** | **N/A** |
| **This system, protocol-exact** | **1.2B + 23M** | **3,000 located words** | **0.3339** | **0.1065** | **0.2930** | **0.8680** | **N/A** |
| gpt-5.6-luna, zero-shot² | undisclosed | full transcript | 0.3243 | 0.0789 | 0.2749 | 0.8608 | $2.05 |
| gpt-5.6-sol, zero-shot² | undisclosed | full transcript | 0.3092 | 0.0694 | 0.2610 | 0.8583 | $10.23 |
| claude-opus-5, zero-shot² | undisclosed | full transcript | 0.2887 | 0.0843 | 0.2501 | 0.8522 | $16.20 |
| claude-haiku-4-5, zero-shot² | undisclosed | full transcript | 0.2866 | 0.0842 | 0.2419 | 0.8431 | $2.35 |
| distilbart-cnn-12-6, community checkpoint³ | 306M | truncated to model context | 0.2865 | 0.0654 | 0.2550 | | N/A |
| claude-sonnet-5, zero-shot² | undisclosed | full transcript | 0.2789 | 0.0753 | 0.2398 | 0.8478 | $6.50 |
| bart-large-cnn, community checkpoint³ | 406M | truncated to model context | 0.2732 | 0.0565 | 0.2408 | | N/A |
| pegasus-cnn, community checkpoint³ | 570M | truncated to model context | 0.2009 | 0.0445 | 0.1723 | | N/A |
| led-base, community checkpoint³ | 162M | truncated to model context | 0.0941 | 0.0237 | 0.0796 | | N/A |

¹ The one row we did not generate: scored from the authors' released prediction files under our
frozen scorer, gated on first reproducing their reported score. Their model reads the
transcript up to its ~11,800-word training-time truncation. We also ran their released
checkpoint end to end through our own harness (`eval/segenc.py` ports their model classes);
that inference scores a stable 3.3 ROUGE-1 below what their released predictions score, an
offset that survived ruling out precision and padding and is likeliest explained by the
uploaded weights differing from the exact seed checkpoint behind their released predictions.
That run ships as `results/rows/segenc-port-full-test.json`, marked calibration-only: quoting
it as their system row would understate the strongest published specialist by an artifact of
our port, so the headline row stays sourced from their own predictions.
² Frontier baselines read the full untruncated transcript (the 40,000-word budget exceeds the
longest transcript in the split), zero-shot, non-reasoning regime, same prompt template as
ours. API cost is the batch price for the full 281-query split, recorded at collect time
(2026-07-27); per-row figures live in `cost_usd_total` in `results/rows/`. N/A means no API
in the loop: those rows run on local GPU time (a full-split run of this system takes about
11 minutes on an RTX 5070 Ti).
³ Query-prefixed transcript truncated to each checkpoint's own encoder context
(`eval/hf_rows.py`).

Three orderings carry the paper:

- **The fine-tuned 1.2B beats all five frontier zero-shot baselines on ROUGE-1 and
  BERTScore**, paired-bootstrap intervals excluding zero, while reading 2,000 located words
  against their full transcript.
- **A 2023 406M specialist beats every 2026 frontier model by 6.2 ROUGE-1** under one scorer.
  It also beats this system by 3.2 ROUGE-1, an interval firmly excluding zero, while BERTScore
  cannot separate the two systems. That metric disagreement is one of the paper's measurement
  findings.
- **The 406M SegEnc, retrained on the same located spans this system reads, is statistically
  indistinguishable from the 1.2B on every metric** (all intervals cross zero) at 2.9x fewer
  parameters and 2.16x less peak inference VRAM. Training regime and architecture outweigh
  scale, which is the paper's title claim.

`results/CHART.md` holds every row the project scored, including the validation sweeps, plus
a separate, deliberately incommensurable table of published numbers from the literature.
Published QMSum numbers use ROUGE implementations the benchmark never specified, so they must
never share a column with these rows; the paper measures the cross-implementation delta.

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
