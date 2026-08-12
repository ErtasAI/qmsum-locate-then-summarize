"""Make beam search correct for LFM2 hybrid conv/attention caches.

Found 2026-07-30 while opening the decode sweep: `model.generate(num_beams=4)` on
LFM2.5-1.2B raises `IndexError: index out of range in self` inside
`Lfm2HybridConvCache.reorder_cache` in transformers 4.57.6.

There are TWO defects in that method and only one of them crashes.

    def reorder_cache(self, beam_idx):
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx]  = self.key_cache[layer_idx].index_select(0, beam_idx)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)
            self.conv_cache[layer_idx]  = self.conv_cache[layer_idx].index_select(0, beam_idx)

**Defect 1, loud.** `update()` only appends to `key_cache` for attention layers, and pads
the conv-layer slots with `torch.tensor([])` placeholders. `index_select` on a 0-row tensor
with beam indices 0..3 raises. LFM2.5-1.2B has 16 layers of which 9 are conv, so this fires
immediately. The same class already guards this exact condition in `get_seq_length`
(`self.key_cache[layer_idx].numel() == 0`), so skipping empties is the file's own idiom.

**Defect 2, silent, and the reason this module exists rather than a try/except.** The loop is
bounded by `len(self.key_cache)`, which only reaches the LAST ATTENTION layer. LFM2.5-1.2B's
layer_types are:

    ['conv','conv','full_attention','conv','conv','full_attention','conv','conv',
     'full_attention','conv','full_attention','conv','full_attention','conv',
     'full_attention','conv']

The last attention layer is index 14, so `len(key_cache) == 15` while `conv_cache` has 16
entries. **`conv_cache[15]` is never reordered.** Layer 15's convolution state would keep
whichever beam happened to occupy that slot before the reorder, so generation continues with
one layer's short-conv history belonging to a different hypothesis. Nothing raises, output
looks fluent, and the numbers are wrong. Any model whose final layer is a conv layer is
affected, which for this family is the common case.

So the fix reorders `conv_cache` over ALL layers and key/value over the populated ones,
rather than sharing one loop bound between two differently-sized structures.

**Why a monkeypatch and not a venv edit.** `.venv` is frozen and its contents are not in git,
so an edit there is invisible to review, lost on any reinstall, and would silently change the
behaviour of every other script in the project. This module is imported explicitly, only on
the beam path, and is covered by tests.

**Why it is not applied unconditionally.** The frozen greedy path produced 281 test
predictions that were verified byte-identical on a re-run, and that guarantee is only worth
having if nothing can perturb the default. `reorder_cache` is never called during greedy
decoding, so patching would be inert, but `run_pipeline` still calls `apply()` only when
`num_beams > 1` so the default path is provably untouched rather than argued to be.

Upstream: worth filing against transformers. Defect 2 in particular is a silent-correctness
bug affecting any hybrid model ending in a non-attention layer.
"""

_APPLIED = False


def _reorder_cache(self, beam_idx):
    """Beam-reorder a hybrid conv/attention cache.

    Reorders conv state for every layer, and key/value state for the layers that have it.
    """
    # Conv state exists for EVERY layer and must be reordered for every layer. Bounding
    # this by len(self.key_cache) is defect 2 and drops the trailing conv layers.
    for layer_idx in range(len(self.conv_cache)):
        conv = self.conv_cache[layer_idx]
        self.conv_cache[layer_idx] = conv.index_select(0, beam_idx.to(conv.device))

    # Key/value exist only for attention layers; conv-layer slots hold empty placeholders.
    for layer_idx in range(len(self.key_cache)):
        key = self.key_cache[layer_idx]
        if key.numel():
            self.key_cache[layer_idx] = key.index_select(0, beam_idx.to(key.device))
        value = self.value_cache[layer_idx]
        if value.numel():
            self.value_cache[layer_idx] = value.index_select(0, beam_idx.to(value.device))


def apply():
    """Patch Lfm2HybridConvCache.reorder_cache. Idempotent; returns True if applied."""
    global _APPLIED
    if _APPLIED:
        return False
    from transformers.models.lfm2.modeling_lfm2 import Lfm2HybridConvCache
    Lfm2HybridConvCache.reorder_cache = _reorder_cache
    _APPLIED = True
    return True


def is_applied():
    return _APPLIED
