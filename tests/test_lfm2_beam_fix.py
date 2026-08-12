"""Tests for the LFM2 hybrid-cache beam-reorder fix.

The important test here is the trailing-conv-layer one. Defect 1 crashes and would be caught
by any smoke run; defect 2 is silent, so only a test pins it.
"""
import torch

from pipeline import lfm2_beam_fix


class FakeHybridCache:
    """Stands in for Lfm2HybridConvCache with LFM2.5-1.2B's actual layer layout.

    Reproduces the two structural properties that cause the bugs: conv-layer slots in
    key_cache hold empty placeholder tensors, and key_cache stops at the last attention
    layer while conv_cache covers all of them.
    """

    LAYER_TYPES = ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                   "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                   "full_attention", "conv", "full_attention", "conv"]

    def __init__(self, batch=4):
        self.batch = batch
        last_attention = max(i for i, t in enumerate(self.LAYER_TYPES)
                             if t == "full_attention")
        self.key_cache, self.value_cache = [], []
        for layer_idx in range(last_attention + 1):
            if self.LAYER_TYPES[layer_idx] == "full_attention":
                # Distinct value per batch row so a reorder is observable.
                self.key_cache.append(torch.arange(batch, dtype=torch.float32)
                                      .reshape(batch, 1, 1, 1).clone())
                self.value_cache.append(torch.arange(batch, dtype=torch.float32)
                                        .reshape(batch, 1, 1, 1).clone())
            else:
                self.key_cache.append(torch.tensor([]))
                self.value_cache.append(torch.tensor([]))
        self.conv_cache = [torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1).clone()
                           for _ in self.LAYER_TYPES]


def test_the_layout_under_test_actually_triggers_both_defects():
    """Guards the fixture: if this stops holding, the tests below prove nothing."""
    c = FakeHybridCache()
    assert len(c.key_cache) == 15
    assert len(c.conv_cache) == 16
    assert c.LAYER_TYPES[-1] == "conv"          # trailing conv layer, i.e. defect 2 is live
    assert any(k.numel() == 0 for k in c.key_cache)  # empty placeholders, i.e. defect 1


def test_reorder_does_not_raise_on_empty_placeholder_layers():
    """Defect 1: the crash we actually hit."""
    c = FakeHybridCache()
    lfm2_beam_fix._reorder_cache(c, torch.tensor([3, 2, 1, 0]))


def test_every_conv_layer_is_reordered_including_the_trailing_one():
    """Defect 2, the silent one. conv_cache[15] lies beyond len(key_cache) and upstream
    never touches it, so layer 15 would keep another hypothesis' convolution state."""
    c = FakeHybridCache()
    lfm2_beam_fix._reorder_cache(c, torch.tensor([3, 2, 1, 0]))
    expected = torch.tensor([3.0, 2.0, 1.0, 0.0]).reshape(4, 1, 1)
    for layer_idx in range(len(c.conv_cache)):
        assert torch.equal(c.conv_cache[layer_idx], expected), f"layer {layer_idx} not reordered"


def test_populated_key_and_value_layers_are_reordered():
    c = FakeHybridCache()
    lfm2_beam_fix._reorder_cache(c, torch.tensor([1, 0, 3, 2]))
    expected = torch.tensor([1.0, 0.0, 3.0, 2.0]).reshape(4, 1, 1, 1)
    for layer_idx, kind in enumerate(FakeHybridCache.LAYER_TYPES[:len(c.key_cache)]):
        if kind == "full_attention":
            assert torch.equal(c.key_cache[layer_idx], expected)
            assert torch.equal(c.value_cache[layer_idx], expected)


def test_empty_placeholders_stay_empty_rather_than_being_filled():
    c = FakeHybridCache()
    lfm2_beam_fix._reorder_cache(c, torch.tensor([0, 1, 2, 3]))
    for layer_idx, kind in enumerate(FakeHybridCache.LAYER_TYPES[:len(c.key_cache)]):
        if kind == "conv":
            assert c.key_cache[layer_idx].numel() == 0
            assert c.value_cache[layer_idx].numel() == 0


def test_a_duplicating_beam_index_is_handled():
    """Beam search routinely selects the same source hypothesis for several beams."""
    c = FakeHybridCache()
    lfm2_beam_fix._reorder_cache(c, torch.tensor([2, 2, 2, 2]))
    assert torch.equal(c.conv_cache[15], torch.full((4, 1, 1), 2.0))


def test_apply_is_idempotent_and_patches_the_real_class():
    from transformers.models.lfm2.modeling_lfm2 import Lfm2HybridConvCache
    original = Lfm2HybridConvCache.reorder_cache
    try:
        lfm2_beam_fix._APPLIED = False
        assert lfm2_beam_fix.apply() is True
        assert lfm2_beam_fix.is_applied()
        assert Lfm2HybridConvCache.reorder_cache is lfm2_beam_fix._reorder_cache
        assert lfm2_beam_fix.apply() is False      # second call is a no-op
    finally:
        Lfm2HybridConvCache.reorder_cache = original
        lfm2_beam_fix._APPLIED = False


def test_greedy_path_is_untouched_unless_apply_is_called():
    """The frozen greedy config reproduced 281 predictions byte-identically. Importing this
    module must not change that; only an explicit apply() may."""
    from transformers.models.lfm2.modeling_lfm2 import Lfm2HybridConvCache
    lfm2_beam_fix._APPLIED = False
    assert Lfm2HybridConvCache.reorder_cache is not lfm2_beam_fix._reorder_cache
