"""Shared price loading and cost accounting for hosted-API baselines.

The guard here is deliberate: unverified or zero prices fail loudly rather than
logging a plausible-looking number into results/rows/*.json, because those
numbers end up in published cost claims.
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

def load_prices(path=None):
    prices = json.loads((pathlib.Path(path) if path else ROOT / "eval" / "prices.json").read_text())
    if not prices.get("_verified_on"):
        raise ValueError(
            "eval/prices.json has no _verified_on date. Confirm every rate you intend to run "
            "against the vendor's own pricing page, then set _verified_on to that date. "
            "The shipped values are secondary-source starting points, not verified rates.")
    return prices

def cost_usd(model, prompt_toks, completion_toks, prices, batch=False):
    """USD for one response. `batch=True` bills at the 50%-discount batch rate.

    Batch rates are a separate column in prices.json rather than a computed
    halving, so a vendor changing the discount cannot silently misprice a run.
    Callers pass the mode recorded on the individual cached response, not the
    mode of the current run, so a cache mixing synchronous and batched entries
    still costs each row at what was actually paid for it.
    """
    p = prices[model]  # KeyError = fail loudly, add the model to prices.json first
    ink, outk = (("input_per_1m_batch", "output_per_1m_batch") if batch
                 else ("input_per_1m", "output_per_1m"))
    if ink not in p or outk not in p:
        raise ValueError(f"prices.json has no batch rates for {model}; add "
                         f"{ink}/{outk} from the vendor's pricing page before batching")
    if p[ink] <= 0 or p[outk] <= 0:
        raise ValueError(f"prices.json has no real prices for {model}; fill it from the live pricing page first")
    return prompt_toks / 1e6 * p[ink] + completion_toks / 1e6 * p[outk]
