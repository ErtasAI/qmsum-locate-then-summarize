"""Tests for the ported SegEnc chunk tokenizer.

The ablation this module exists for is a within-port comparison, so a chunking error would not
show up as a crash or an obviously wrong score. It would show up as a plausible number, which is
the failure mode worth testing against. Every expectation here is read off
`external/query-focused-sum/multiencoder/dataset.py`.

No model is loaded: these test the tokenisation contract only, so they run without a GPU.
"""
import pytest
import torch
from transformers import BartTokenizerFast

from eval.segenc import ChunkTokenizer, CHUNK_SIZE, MAX_NUM_CHUNKS


@pytest.fixture(scope="module")
def tok():
    return BartTokenizerFast.from_pretrained("facebook/bart-large")


def _source(n_words):
    return " ".join(f"word{i}" for i in range(n_words))


def test_stride_yields_max_chunks_plus_max_chunks_minus_one(tok):
    """Their stride emits a second chunk set with max_num_chunks - 1 chunks. At the published
    settings that is 32 + 31 = 63, which is the number that drives the memory cost."""
    ct = ChunkTokenizer(tok, max_num_chunks=MAX_NUM_CHUNKS, stride=True, pad=True)
    out = ct(_source(30000), query="what did they decide")
    assert out["input_ids"].shape[0] == MAX_NUM_CHUNKS + (MAX_NUM_CHUNKS - 1) == 63


def test_no_stride_yields_exactly_max_chunks(tok):
    ct = ChunkTokenizer(tok, max_num_chunks=8, stride=False, pad=True)
    out = ct(_source(30000), query="q")
    assert out["input_ids"].shape[0] == 8


def test_every_chunk_is_chunk_size_wide(tok):
    ct = ChunkTokenizer(tok, max_num_chunks=8, stride=True, pad=True)
    out = ct(_source(30000), query="what did they decide")
    assert out["input_ids"].shape[1] == CHUNK_SIZE
    assert out["attention_mask"].shape == out["input_ids"].shape


def test_the_query_prefix_is_repeated_on_every_chunk(tok):
    """The property that makes padding non-cosmetic: the query is re-encoded per chunk."""
    query = "what did the committee decide about funding"
    ct = ChunkTokenizer(tok, max_num_chunks=6, stride=False, pad=True)
    out = ct(_source(30000), query=query)
    prefix_len = tok(f"<s>{query}</s>", add_special_tokens=False,
                     return_tensors="pt")["input_ids"].size(-1)
    first = out["input_ids"][0, :prefix_len]
    for row in range(out["input_ids"].shape[0]):
        assert torch.equal(out["input_ids"][row, :prefix_len], first)
    # And the prefix is always attended, even on chunks whose text half is all padding.
    assert out["attention_mask"][:, :prefix_len].min().item() == 1


def test_pad_true_holds_chunk_count_constant_across_input_sizes(tok):
    """This is what makes pad=True the right setting for the ablation: chunk count is fixed, so
    the only variable between the full-transcript arm and the spans arm is source content."""
    ct = ChunkTokenizer(tok, max_num_chunks=16, stride=True, pad=True)
    short = ct(_source(50), query="q")["input_ids"].shape[0]
    long = ct(_source(30000), query="q")["input_ids"].shape[0]
    assert short == long == 31


def test_pad_false_lets_chunk_count_track_input_size(tok):
    """The behaviour that would confound the ablation if pad were False."""
    ct = ChunkTokenizer(tok, max_num_chunks=16, stride=True, pad=False)
    short = ct(_source(50), query="q")["input_ids"].shape[0]
    long = ct(_source(30000), query="q")["input_ids"].shape[0]
    assert short < long


def test_source_is_truncated_to_the_chunk_budget(tok):
    """SegEnc does not see an unbounded transcript: the source is cut to
    max_num_chunks * (chunk_size - prefix_len) tokens, about 11,800 words at their settings."""
    query = "q"
    ct = ChunkTokenizer(tok, max_num_chunks=4, stride=False, pad=False)
    out = ct(_source(30000), query=query)
    prefix_len = tok(f"<s>{query}</s>", add_special_tokens=False,
                     return_tensors="pt")["input_ids"].size(-1)
    suffix_chunk = CHUNK_SIZE - prefix_len
    real = int(out["attention_mask"][:, prefix_len:].sum().item())
    assert real <= 4 * suffix_chunk


def test_padding_positions_are_masked_out(tok):
    ct = ChunkTokenizer(tok, max_num_chunks=16, stride=False, pad=True)
    out = ct(_source(30), query="q")
    assert out["attention_mask"].min().item() == 0
    # The text half of a mostly-empty input must contain masked positions.
    assert out["attention_mask"][:, 1:].min().item() == 0


def test_works_without_a_query(tok):
    ct = ChunkTokenizer(tok, max_num_chunks=4, stride=False, pad=True)
    out = ct(_source(2000), query=None)
    assert out["input_ids"].shape[0] == 4
    assert out["input_ids"].shape[1] == CHUNK_SIZE


def test_published_settings_are_the_module_defaults():
    """These come from multiencoder/README.md and the checkpoint's own config.json. A change
    here silently changes what 'their regime' means in the ablation."""
    assert CHUNK_SIZE == 512
    assert MAX_NUM_CHUNKS == 32
