# QMSum challenge chart

| run_id | split | n | R1 | R2 | RLsum | BERTScore | cost USD | s/query | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| row-socratic-segenc-test | test | 281 | 0.386 | 0.139 | 0.337 | 0.874 |  |  |  |
| segenc-c8cont-e4-test | test | 281 | 0.363 | 0.127 | 0.322 | 0.871 | 0.00 | 1.9 | 2.65 |
| loc-w375-l12-b2000-test | test | 281 | 0.354 | 0.123 | 0.314 | 0.873 | 0.00 | 2.4 |  |
| segenc-port-full-test | test | 281 | 0.353 | 0.119 | 0.306 | 0.870 | 0.00 | 4.6 | 7.96 |
| m3-fusion-lfm2.5-1.2b-test | test | 281 | 0.334 | 0.107 | 0.293 | 0.868 | 0.00 | 2.6 |  |
| baseline-gpt-5.6-luna-test | test | 281 | 0.324 | 0.079 | 0.275 | 0.861 | 2.05 |  |  |
| baseline-gpt-5.6-sol-test | test | 281 | 0.309 | 0.069 | 0.261 | 0.858 | 10.23 |  |  |
| zeroshot-lfm25-spans-test | test | 281 | 0.301 | 0.067 | 0.250 | 0.869 |  | 1.2 | 5.59 |
| baseline-claude-opus-5-test | test | 281 | 0.289 | 0.084 | 0.250 | 0.852 | 16.20 |  |  |
| baseline-claude-haiku-4-5-test | test | 281 | 0.287 | 0.084 | 0.242 | 0.843 | 2.35 |  |  |
| row-distilbart-test | test | 281 | 0.286 | 0.065 | 0.255 | 0.855 | 0.00 |  |  |
| zeroshot-lfm25-trunc-test | test | 281 | 0.286 | 0.056 | 0.245 | 0.861 |  | 2.0 | 15.08 |
| baseline-claude-sonnet-5-test | test | 281 | 0.279 | 0.075 | 0.240 | 0.848 | 6.50 |  |  |
| row-bart-large-cnn-test | test | 281 | 0.273 | 0.056 | 0.241 | 0.851 | 0.00 |  |  |
| row-pegasus-test | test | 281 | 0.201 | 0.044 | 0.172 | 0.834 | 0.00 |  |  |
| row-led-base-test | test | 281 | 0.094 | 0.024 | 0.080 | 0.782 | 0.00 |  |  |
| row-socratic-segenc-val | val | 272 | 0.386 | 0.134 | 0.340 |  |  |  |  |
| segenc-c8cont-e1-val | val | 272 | 0.383 | 0.129 | 0.335 |  | 0.00 | 1.9 |  |
| segenc-spansft-e4-val | val | 272 | 0.375 | 0.124 | 0.327 |  | 0.00 | 5.4 |  |
| segenc-c8-e4-val | val | 272 | 0.371 | 0.121 | 0.324 |  | 0.00 | 1.7 |  |
| segenc-c8cont-e4-val-vram | val | 272 | 0.371 | 0.122 | 0.323 |  | 0.00 | 1.9 | 2.65 |
| segenc-c8cont-e4-val | val | 272 | 0.371 | 0.122 | 0.323 |  | 0.00 | 1.8 |  |
| segenc-c8cont-e3-val | val | 272 | 0.370 | 0.119 | 0.322 |  | 0.00 | 1.8 |  |
| m2-goldspans-val | val | 237 | 0.369 | 0.128 | 0.325 | 0.874 | 0.00 |  |  |
| lfm25-goldspans-val | val | 237 | 0.367 | 0.128 | 0.324 |  | 0.00 |  |  |
| segenc-c8cont-e2-val | val | 272 | 0.366 | 0.117 | 0.318 |  | 0.00 | 1.8 |  |
| segenc-spansft-e2-val | val | 272 | 0.366 | 0.122 | 0.322 |  | 0.00 | 5.0 |  |
| segenc-spansft-e3-val | val | 272 | 0.358 | 0.118 | 0.312 |  | 0.00 | 4.8 |  |
| loc-w375-l12-b2000-val-vram | val | 272 | 0.354 | 0.114 | 0.311 |  | 0.00 | 2.8 | 5.73 |
| loc-w375-l12-b2000-val | val | 272 | 0.354 | 0.114 | 0.311 | 0.872 | 0.00 | 2.4 |  |
| segenc-port-full-val | val | 272 | 0.353 | 0.114 | 0.311 |  | 0.00 | 4.9 |  |
| segenc-spansft-e1-val | val | 272 | 0.352 | 0.122 | 0.312 |  | 0.00 | 4.4 |  |
| segenc-c8-e2-val | val | 272 | 0.351 | 0.116 | 0.309 |  | 0.00 | 1.4 |  |
| loc-w375-l12-b1500-val | val | 272 | 0.347 | 0.111 | 0.308 | 0.873 | 0.00 | 2.1 |  |
| loc-w375-l12-val | val | 272 | 0.345 | 0.103 | 0.304 | 0.871 | 0.00 | 2.8 |  |
| gguf-q4km-val | val | 272 | 0.343 | 0.107 | 0.303 |  | 0.00 | 0.4 |  |
| sweep-decode-b2nr3-val | val | 272 | 0.340 | 0.087 | 0.298 | 0.870 | 0.00 | 3.6 |  |
| m3-fusion-lfm2.5-1.2b-val | val | 272 | 0.339 | 0.101 | 0.299 | 0.869 | 0.00 |  |  |
| m3-lfm25-seed101-val | val | 272 | 0.338 | 0.104 | 0.299 | 0.870 | 0.00 | 2.2 | 5.73 |
| m2-budget2000-val | val | 272 | 0.336 | 0.101 | 0.295 | 0.866 | 0.00 |  |  |
| sweep-decode-b2-val | val | 272 | 0.335 | 0.101 | 0.294 | 0.866 | 0.00 | 3.9 |  |
| m2-fusion-qwen3-1.7b-val | val | 272 | 0.334 | 0.098 | 0.294 | 0.866 | 0.00 |  |  |
| sweep-decode-nr3-val | val | 272 | 0.333 | 0.085 | 0.294 | 0.868 | 0.00 | 2.5 |  |
| m3-fusion-locspan-qwen3-1.7b-val | val | 272 | 0.329 | 0.099 | 0.290 | 0.865 | 0.00 |  |  |
| m2-budget4000-val | val | 272 | 0.324 | 0.098 | 0.285 | 0.863 | 0.00 |  |  |
| dialogled-large-fusion-val | val | 272 | 0.307 | 0.073 | 0.268 | 0.854 | 0.00 |  |  |
| segenc-port-spans-val | val | 272 | 0.290 | 0.095 | 0.255 |  | 0.00 | 3.2 |  |
| zeroshot-lfm25-spans-val | val | 272 | 0.290 | 0.060 | 0.244 | 0.866 |  | 1.2 | 5.70 |
| zeroshot-lfm25-trunc-val | val | 272 | 0.287 | 0.050 | 0.246 | 0.860 |  | 2.1 | 16.57 |
| m1-qwen3-1.7b-val | val | 272 | 0.227 | 0.062 | 0.194 | 0.830 | 0.00 |  |  |

## Cited numbers (different protocols, not comparable to our rows)

| Source | Model | Number | Protocol flag |
|---|---|---|---|
| ZeroSCROLLS | GPT-4 | 18.5 | geometric mean R-1/2/L, zero-shot |
| ZeroSCROLLS | Claude (v1 era) | 14.6 | geometric mean R-1/2/L, zero-shot |
| LongBench v1 qmsum | GPT-3.5-Turbo-16k | 23.4 | ROUGE-L, subset variant |
| DialogLM paper (AAAI'22) Tab.8 | **DialogLED-large** (fine-tuned) | 34.50 R-1 | full test, own fine-tune |
| DialogLM paper (AAAI'22) Tab.3 | DialogLM base, dense | 34.02 R-1 | full test; NOT DialogLED, previously mislabelled here |
| DialogLM paper (AAAI'22) Tab.3 | DialogLM base, sparse | 33.69 R-1 | full test; the row Liu and Xu quote |
| QMSum paper | HMNet + gold spans | 36.51 R-1 | oracle locator |
