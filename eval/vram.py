"""Peak-VRAM instrumentation, shared by our pipeline and the SegEnc port.

Measuring this in two places under two conventions would produce two numbers that cannot
be compared, which defeats the reason for recording it at all: the head-to-head table
needs a 1.2B decoder-only pipeline and a 406M encoder-decoder measured the same way, on
the same queries, on the same card.

Three conventions are fixed here so neither call site can drift.

**Allocated is the headline; reserved is recorded beside it.** `max_memory_allocated`
counts live tensor bytes. `max_memory_reserved` counts what the caching allocator holds
from the driver, which is larger, depends on allocation history, and is roughly what
`nvidia-smi` shows. A paper comparing two architectures wants the first. A practitioner
asking "what card do I need" wants the second. Reporting only one invites the reader to
assume it was the other.

**Peak is per query, reset before each.** One reset at process start would give a single
number for the run and hide the distribution. Resetting per query yields the same
headline (the max over queries) plus a median and a p95, and makes each row independent
of the order queries happened to arrive in.

**Weights are inside the number.** The peak is taken against a model already resident, so
a row reads as the total a card must hold rather than as an activation delta. This is the
convention the earlier `eval/beam_memory_probe.py` and `eval/segenc.py --probe` runs used,
so figures recorded earlier in the project stay comparable to anything measured through here.

On a machine with no CUDA device every function is a no-op returning None, so importing
this module never forces a GPU and the tests run anywhere.
"""

BYTES_PER_GB = 1e9


def _torch():
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep everywhere we run
        return None
    return torch if torch.cuda.is_available() else None


def available():
    """True when peaks are measurable. Call sites should record None, not 0.0, when False."""
    return _torch() is not None


def reset():
    """Start a fresh peak window. Safe to call on CPU."""
    torch = _torch()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()


def peak_gb():
    """Peak ALLOCATED gigabytes since the last `reset()`, or None without CUDA."""
    torch = _torch()
    if torch is None:
        return None
    return round(torch.cuda.max_memory_allocated() / BYTES_PER_GB, 3)


def peak_reserved_gb():
    """Peak RESERVED gigabytes since the last `reset()`, or None without CUDA.

    Always larger than `peak_gb()`. Quote this one when the claim is about what fits on
    a card, and the allocated figure when the claim is about a model's memory cost.
    """
    torch = _torch()
    if torch is None:
        return None
    return round(torch.cuda.max_memory_reserved() / BYTES_PER_GB, 3)


def record():
    """Both peaks as a dict ready to merge into a prediction record.

    Returns an EMPTY dict without CUDA rather than a dict of Nones, so a CPU run writes
    preds files that are byte-identical to what it wrote before this module existed.
    """
    if not available():
        return {}
    return {"peak_vram_gb": peak_gb(), "peak_vram_reserved_gb": peak_reserved_gb()}


def summarize(values):
    """max / median / p95 over per-query peaks, or None if nothing was measured.

    The max is the number that decides which card runs the system, so it leads. p95 is
    reported because this task's memory is driven by input length, whose distribution is
    long-tailed: a median hides the query that decides the requirement.
    """
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    return {
        "max": vals[-1],
        "median": vals[n // 2] if n % 2 else round((vals[n // 2 - 1] + vals[n // 2]) / 2, 3),
        "p95": vals[min(n - 1, int(round(0.95 * (n - 1))))],
        "min": vals[0],
        "n": n,
    }
