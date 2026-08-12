"""Run the released Socratic SegEnc checkpoint ourselves, under our protocol.

WHY, and it is not primarily to fine-tune it. `results/rows/row-socratic-segenc-test.json`
(R1 0.3860) was produced by scoring the authors' PUBLISHED predictions with our scorer. That is
clean but it buys only one row and no ablations. Loading the checkpoint ourselves buys three
things we cannot get otherwise:

1. **An independent check on that row.** If our own inference reproduces roughly 0.386, the
   published-predictions row is confirmed by a second route.
2. **THE ablation the paper's outline originally conceded it could not do.** It conceded
   we "cannot say whether its advantage comes from full-transcript encoding, from Socratic
   pretraining, or from BART-large's summarization prior". Feeding this checkpoint OUR located
   spans instead of the full transcript separates the first from the other two directly. If the
   score collapses toward ours, the advantage is the input regime; if it holds, it is the model.
   That is the single most informative experiment left in the project and it is inference-only.
3. A starting point for continue-fine-tuning, if that is still wanted afterwards.

ATTRIBUTION. `BartForMultiConditionalGeneration` and `ChunkTokenizer` below are ports of
`multiencoder/models.py` and `multiencoder/dataset.py` from `salesforce/query-focused-sum`,
Copyright (c) 2021 salesforce.com inc., **BSD-3-Clause** (`external/query-focused-sum/LICENSE.txt`,
retained in-repo). The architecture is Fusion-in-Decoder over a single long document: chunks are
encoded independently and their encoder states concatenated so the decoder cross-attends over all
of them. Reproduced rather than imported because their repo pins a transformers generation whose
torch build has no kernels for this GPU.

THE CHUNKING DETAILS THAT ARE EASY TO GET WRONG, all verified against their source:
- The query is re-prepended to EVERY chunk as `<s>{query}</s>`, and the per-chunk text budget is
  `chunk_size - len(prefix)`, so the query is paid for once per chunk.
- The source is truncated to `max_num_chunks * suffix_chunk_size` tokens, roughly 15,700 at their
  published settings, i.e. about 11,800 words. **SegEnc does not see a whole long transcript
  either**; it sees about 4x what our 3,000-word budget admitted.
- `stride` emits a SECOND chunk set offset by half a chunk, with `max_num_chunks - 1` chunks, and
  concatenates it. At their settings that is 32 + 31 = **63 chunks of 512 tokens, 32,256 encoder
  positions**. It re-covers the same token range with shifted boundaries, so it doubles compute
  without extending coverage.
"""
import argparse
import json
import math
import pathlib
import time

import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers.modeling_outputs import BaseModelOutput

from eval import vram

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Their published predict settings (multiencoder/README.md).
CHUNK_SIZE = 512
MAX_NUM_CHUNKS = 32
STRIDE = True
GENERATION_MAX_LEN = 256
NUM_BEAMS = 4              # from the checkpoint's own config.json
NO_REPEAT_NGRAM_SIZE = 3   # ditto
EARLY_STOPPING = True      # ditto


class BartForMultiConditionalGeneration(BartForConditionalGeneration):
    """Fusion-in-Decoder BART. Port of salesforce/query-focused-sum, BSD-3-Clause."""

    def multi_encode(self, input_ids=None, attention_mask=None, return_dict=True):
        # (B, N, L) -> (B*N, L) -> (B*N, L, D) -> (B, N*L, D)
        B, N, L = input_ids.size(0), input_ids.size(1), input_ids.size(2)
        if input_ids.size() != attention_mask.size():
            raise ValueError(f"input_ids {input_ids.size()} != attention_mask {attention_mask.size()}")
        flat_ids = input_ids.contiguous().view(B * N, L)
        flat_mask = attention_mask.contiguous().view(B * N, L)
        encoder_outputs = self.model.encoder(input_ids=flat_ids, attention_mask=flat_mask,
                                             return_dict=True)
        hidden = encoder_outputs.last_hidden_state
        stacked = hidden.contiguous().view(B, N * L, hidden.size(2))
        stacked_mask = flat_mask.contiguous().view(B, N * L)
        return BaseModelOutput(last_hidden_state=stacked), stacked_mask

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        encoder_outputs, attention_mask = self.multi_encode(input_ids, attention_mask)
        return super().generate(input_ids=None, attention_mask=attention_mask,
                                encoder_outputs=encoder_outputs, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None, **kwargs):
        if input_ids is None:
            if encoder_outputs is None:
                raise ValueError("encoder_outputs is required when no input_ids are passed")
        else:
            encoder_outputs, attention_mask = self.multi_encode(input_ids, attention_mask)
        return super().forward(input_ids=None, attention_mask=attention_mask,
                               encoder_outputs=encoder_outputs, **kwargs)


class ChunkTokenizer:
    """Port of their ChunkTokenizer. Chunking happens on TOKENS, not words."""

    # pad defaults to True because that is their EFFECTIVE setting: `ChunkTokenizer`'s own default
    # is False, but `MultiEncoderDataset.__init__` defaults `pad=True` and passes it through, and
    # the dataset is what their training and predict paths construct.
    #
    # This is not a cosmetic difference, which is why it gets a comment. The query prefix is
    # prepended to EVERY chunk with attention 1, including chunks whose text half is entirely
    # padding. So with pad=True the decoder always cross-attends over 63 copies of the query, and
    # with pad=False it sees only as many copies as there are real chunks. Running with pad=False
    # gave 63 chunks on 5 of 30 val queries and fewer on the other 25, and reproduced their
    # published predictions at only 0.534 mean token F1 with outputs 15 words short.
    def __init__(self, tokenizer, chunk_size=CHUNK_SIZE, max_num_chunks=MAX_NUM_CHUNKS,
                 stride=STRIDE, pad=True):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_num_chunks = max_num_chunks
        self.stride = stride
        self.pad = pad

    def __call__(self, source, query=None):
        prefix = f"<s>{query}</s>" if query else "<s>"
        prefix_tokens = self.tokenizer(prefix, add_special_tokens=False, return_tensors="pt",
                                       max_length=self.chunk_size, truncation=True)["input_ids"]
        prefix_len = prefix_tokens.size(-1)
        suffix_chunk_size = self.chunk_size - prefix_len
        suffix_total_size = self.max_num_chunks * suffix_chunk_size
        input_ids = self.tokenizer(f"{source}</s>", add_special_tokens=False, truncation=True,
                                   max_length=suffix_total_size)["input_ids"]

        ids_all, mask_all = [], []
        offsets = [False, True] if (self.stride and self.max_num_chunks > 1) else [False]
        for use_offset in offsets:
            if use_offset:
                offset = math.floor(suffix_chunk_size / 2)
                num_chunks = self.max_num_chunks - 1
                suffix_tokens = input_ids[offset: offset + num_chunks * suffix_chunk_size]
            else:
                num_chunks = self.max_num_chunks
                suffix_tokens = input_ids[:suffix_total_size]
            suffix_tokens = list(suffix_tokens)
            suffix_attention = [1] * len(suffix_tokens)
            if self.pad:
                pad_length = max(num_chunks * suffix_chunk_size - len(suffix_tokens), 0)
            else:
                remainder = len(suffix_tokens) % suffix_chunk_size
                pad_length = 0 if remainder == 0 else suffix_chunk_size - remainder
            suffix_tokens += [self.tokenizer.pad_token_id] * pad_length
            suffix_attention += [0] * pad_length
            if not suffix_tokens:
                continue
            suffix_chunks = torch.tensor(suffix_tokens).view(-1, suffix_chunk_size)
            suffix_mask = torch.tensor(suffix_attention).view(-1, suffix_chunk_size)
            prefix_chunks = prefix_tokens.expand(suffix_chunks.size(0), -1)
            ids_all.append(torch.cat((prefix_chunks, suffix_chunks), dim=1))
            mask_all.append(torch.cat((torch.ones_like(prefix_chunks), suffix_mask), dim=1))
        return {"input_ids": torch.cat(ids_all, dim=0),
                "attention_mask": torch.cat(mask_all, dim=0)}


def load(model_dir, dtype=torch.float32, device="cuda"):
    """BART-large is a 2019-era fp32 checkpoint; load fp32 by default and let the caller opt into
    bf16. Pegasus already burned us once on fp16 numerical instability in this repo."""
    tok = BartTokenizerFast.from_pretrained(model_dir)
    model = BartForMultiConditionalGeneration.from_pretrained(model_dir, dtype=dtype)
    model.to(device).eval()
    return tok, model


def build_source(split, mode, span_budget=None, chunk_words=None, locator_ckpt=None):
    """Return {query_id: (query, source_text)} under one of two input regimes.

    mode 'full'  : the whole transcript, i.e. their published regime.
    mode 'spans' : OUR located spans, i.e. the ablation that isolates the input regime.
    """
    from data.normalize import load_meetings, load_queries
    from eval import protocol
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
    out = {}
    if mode == "spans":
        from pipeline.chunker import windows
        from pipeline import locator_crossencoder as lc
        if locator_ckpt:
            lc.CKPT, lc._MODEL = pathlib.Path(locator_ckpt), None
        budget = span_budget or protocol.SPAN_BUDGET_WORDS
        for q in load_queries(split):
            utts = meetings[q["meeting_id"]]
            picked = lc.select(q["query"], windows(utts, chunk_words), budget)
            text = protocol.truncate_words("\n".join(p["text"] for p in picked), budget)
            out[q["query_id"]] = (q["query"], text)
    else:
        for q in load_queries(split):
            out[q["query_id"]] = (q["query"],
                                  protocol.format_transcript(meetings[q["meeting_id"]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="results/external/socratic-qmsum")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--mode", default="full", choices=["full", "spans"],
                    help="'full' reproduces their regime; 'spans' feeds our located spans")
    ap.add_argument("--span-budget", type=int, help="spans mode only")
    ap.add_argument("--chunk-words", type=int, help="spans mode only, locator window size")
    ap.add_argument("--locator-ckpt", help="spans mode only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-num-chunks", type=int, default=MAX_NUM_CHUNKS)
    ap.add_argument("--no-pad", action="store_true",
                    help="do NOT pad to max_num_chunks. Diverges from their setup; see the "
                         "ChunkTokenizer comment on why padding changes the output")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="report peak memory and timing on the longest inputs, write nothing")
    a = ap.parse_args()

    tok, model = load(a.model_dir, dtype=torch.bfloat16 if a.bf16 else torch.float32)
    chunker = ChunkTokenizer(tok, max_num_chunks=a.max_num_chunks, pad=not a.no_pad)
    sources = build_source(a.split, a.mode, a.span_budget, a.chunk_words, a.locator_ckpt)
    items = list(sources.items())

    if a.probe:
        # Longest inputs first: a probe on the first few queries is what hid the beam-search
        # memory cliff earlier today.
        items.sort(key=lambda kv: -len(kv[1][1]))
        items = items[:3]
        print(f"probing {len(items)} longest inputs, max_num_chunks={a.max_num_chunks}, "
              f"dtype={'bf16' if a.bf16 else 'fp32'}", flush=True)
    else:
        items = items[:a.limit] if a.limit else items

    out_path = pathlib.Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with (out_path.open("w", encoding="utf-8") if not a.probe else _null()) as f:
        for i, (qid, (query, source)) in enumerate(items):
            enc = chunker(source, query)
            ids = enc["input_ids"].unsqueeze(0).to(model.device)
            mask = enc["attention_mask"].unsqueeze(0).to(model.device)
            vram.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gen = model.generate(input_ids=ids, attention_mask=mask,
                                 max_length=GENERATION_MAX_LEN, num_beams=NUM_BEAMS,
                                 no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                                 early_stopping=EARLY_STOPPING)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak = vram.peak_gb()
            text = tok.decode(gen[0], skip_special_tokens=True).strip()
            if a.probe:
                print(f"  {qid:12} chunks {ids.shape[1]:3} src_words {len(source.split()):6} "
                      f"{elapsed:7.2f}s  peak {peak:5.2f}GB  out_words {len(text.split())}",
                      flush=True)
            else:
                # Recorded per query rather than only under --probe, so the head-to-head
                # table can quote a distribution over the whole split for both systems
                # instead of a worst case over three hand-picked inputs.
                f.write(json.dumps({"query_id": qid, "prediction": text,
                                    "latency_s": round(elapsed, 4),
                                    "n_chunks": int(ids.shape[1]),
                                    **vram.record()}) + "\n")
                written += 1
                if i % 25 == 0:
                    print(f"  {i}/{len(items)}", flush=True)
            del ids, mask, gen
            torch.cuda.empty_cache()
    if not a.probe:
        print(f"wrote {out_path} ({written} predictions)")


class _null:
    def __enter__(self):
        return None
    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
