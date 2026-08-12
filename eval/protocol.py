"""Frozen eval protocol. Locked 2026-07-23; changing any value invalidates cached rows."""

SEED = 20260723
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
M1_TRUNCATE_WORDS = 4500       # ~6k tokens; reduced from 6000 on 2026-07-23: 8192-ctx QLoRA overflows 16GB VRAM into sysmem fallback
CHUNK_WORDS = 900              # stage-2 window budget
SPAN_BUDGET_WORDS = 3000       # located-span budget fed to the summarizer
JUDGE_SAMPLE_N = 60
# Zero-shot baseline transcript budget. Raised 14000 -> 40000 on 2026-07-27, BEFORE
# any baseline had ever been run (results/baseline_cache/ was empty), so no cached row
# is invalidated. 40000 exceeds the longest transcript in either split (test 25,244
# words; val 24,573), so the frontier baselines now see the FULL transcript, untruncated.
# Rationale: at 14000 the cap bit on 31% of test queries (mean 2,975 words cut, worst
# 11,244). Against million-token context windows that is a handicap on the baseline, and
# the paper's claim is cost-per-quality against a frontier model given every advantage.
# Our own system still sees only SPAN_BUDGET_WORDS of located text; that asymmetry is
# the point of the comparison, not a flaw in it.
PROMPT_WORD_BUDGET = 40000

PROMPT_TEMPLATE = (
    "You are given a meeting transcript and a query. Answer the query with a "
    "concise summary based only on the transcript.\n\n"
    "Query: {query}\n\nTranscript:\n{transcript}\n\nSummary:"
)

def format_transcript(utterances):
    return "\n".join(f"{u['speaker']}: {u['text']}" for u in utterances)

def truncate_words(text, budget):
    words = text.split()
    return text if len(words) <= budget else " ".join(words[:budget])
