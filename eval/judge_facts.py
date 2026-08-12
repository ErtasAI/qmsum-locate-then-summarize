"""Fact-level judging: precision and recall in fact space, not n-gram space.

WHY. Pairwise LLM judging failed as an instrument on this task. Reference-free,
the judge rejects QMSum's own gold answer 54-6 in favour of a frontier summary;
grounding it in the reference moves the verdict barely at all. The decomposition
in eval/precision_recall.py says why: the frontier baselines buy recall with
length (Claude Opus 5, recall 0.629 at precision 0.195, 3.8x the reference
length), and a holistic judge tracks recall. ROUGE-1 F penalises that trade;
ROUGE also can't see a paraphrase.

This measures the same two quantities as ROUGE precision/recall but over
propositions instead of tokens, so a correct paraphrase counts and padding does
not:

    fact recall    = reference facts the summary covers / reference facts
    fact precision = summary claims supported by the reference / summary claims

THREE PROPERTIES THAT MAKE IT FAIRER THAN THE PAIRWISE PANEL.

1. The fact list is extracted from the query and the reference ALONE. The
   extractor never sees a system summary, so the target cannot be shaped by a
   contestant, and every system is scored against a byte-identical list.
2. Scoring is pointwise, not pairwise. There is no A/B ordering, so position
   bias does not exist here rather than being cancelled by swapping.
3. Length cannot buy score. Adding unsupported material lowers precision while
   leaving recall untouched, which is exactly the trade the pairwise judge was
   blind to.

Self-preference still applies: a GPT judge marking GPT summaries is scoring its
own vendor. Both vendors' flagships mark every system, and the per-system gap
between them is reported the same way the panel reports it.

    python -m eval.judge_facts run
    python -m eval.judge_facts run --judges gpt --systems m3-fusion-lfm2.5-1.2b
    python -m eval.judge_facts summarize
"""
import argparse, itertools, json, pathlib, re

from data.normalize import load_queries
from eval import judge as J
from eval import protocol
from eval import run_judge_panel as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "judge_facts"
CACHE_ROOT = OUT_DIR / "cache"

EXTRACT_MAX_TOKENS = 1500
MARK_MAX_TOKENS = 2000

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def build_extract_prompt(query, reference):
    """Decompose the reference. Deliberately shown no summary from any system."""
    return (
        "Below is a query about a meeting and a human-written reference answer. "
        "Break the reference answer into atomic facts: each one a single, "
        "self-contained claim, stated in your own words, containing exactly one "
        "piece of information. Do not add anything that is not in the reference. "
        "Do not merge two claims into one item.\n\n"
        "Reply with JSON only, in this form:\n"
        '{"facts": ["first atomic fact", "second atomic fact"]}\n\n'
        f"Query: {query}\n\nReference answer:\n{reference}")


def build_mark_prompt(query, facts, summary):
    """Score one anonymous summary against the fact list. No system identity,
    no other summary, no ordering: nothing here can carry a contestant's name."""
    numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
    return (
        "Below is a query about a meeting, a numbered list of facts from a "
        "human-written reference answer, and one anonymous summary.\n\n"
        "Do two things.\n"
        "(a) List the numbers of the reference facts the summary conveys. Count a "
        "fact as conveyed if the summary states it in any wording, including "
        "paraphrase. Do not count a fact merely alluded to.\n"
        "(b) Break the summary itself into atomic claims, then list the numbers of "
        "those claims that are NOT supported by the reference facts above.\n\n"
        "Reply with JSON only, in this form:\n"
        '{"covered_fact_numbers": [1, 3], '
        '"summary_claims": ["first claim", "second claim"], '
        '"unsupported_claim_numbers": [2]}\n\n'
        f"Query: {query}\n\nReference facts:\n{numbered}\n\nSummary:\n{summary}")


def parse_json_block(text):
    """Parse the JSON object out of a reply. Returns None rather than a default
    on failure: a silently-empty parse would read as a summary covering nothing."""
    if not text:
        return None
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, dict) else None


def _call(provider, client, model, prompt, max_tokens, cache_file):
    if cache_file and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    resp = J.call_with_retry(J.JUDGES[provider], client, model, prompt,
                             max_tokens=max_tokens)
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(resp), encoding="utf-8")
    return resp


def extract_facts(client, provider, model, split, sample, cache_dir=None, quiet=True):
    """Fact list per query. One canonical list, reused for every system."""
    queries = {q["query_id"]: q for q in load_queries(split)}
    out, failed = {}, []
    for qid in sample:
        q = queries[qid]
        resp = _call(provider, client, model,
                     build_extract_prompt(q["query"], q["reference"]),
                     EXTRACT_MAX_TOKENS,
                     pathlib.Path(cache_dir) / f"{qid}.json" if cache_dir else None)
        parsed = parse_json_block(resp["text"])
        facts = parsed.get("facts") if parsed else None
        if not isinstance(facts, list) or not facts:
            failed.append(qid); continue
        out[qid] = [str(f) for f in facts]
    if failed and not quiet:
        print(f"          fact extraction failed on {len(failed)} queries: {failed[:5]}", flush=True)
    return out, failed


def score_marking(parsed, n_facts):
    """Fact precision and recall from one marking reply.

    A summary the judge decomposes into zero claims gets precision None, not 1.0:
    nothing was verified, and a perfect score for an unparseable answer would be
    the single easiest way for this metric to flatter a system.
    """
    covered = {int(x) for x in parsed.get("covered_fact_numbers", [])
               if isinstance(x, (int, float, str)) and str(x).strip().lstrip("-").isdigit()}
    covered = {c for c in covered if 1 <= c <= n_facts}
    claims = parsed.get("summary_claims") or []
    unsupported = {int(x) for x in parsed.get("unsupported_claim_numbers", [])
                   if isinstance(x, (int, float, str)) and str(x).strip().lstrip("-").isdigit()}
    unsupported = {u for u in unsupported if 1 <= u <= len(claims)}
    recall = len(covered) / n_facts if n_facts else None
    precision = (len(claims) - len(unsupported)) / len(claims) if claims else None
    # INCOHERENT: the judge says the summary conveys reference facts, and in the
    # same reply says not one of its claims is supported by those facts. The two
    # answers are not strictly contradictory (conveying a fact is not the same
    # relation as being entailed by it), but they cannot both feed a meaningful
    # F1, because F1 assumes its two inputs describe one matching relation.
    #
    # Root-caused 2026-07-29: precision's DENOMINATOR is judge-constructed. The
    # fact list behind recall is extracted once from the reference alone and
    # cached, so every system is scored against a byte-identical denominator. The
    # claim list behind precision is rebuilt inside every marking call with no
    # specified granularity, and the two judges split the same frontier summary
    # into 15.6-19.3 vs 10.9-15.0 claims. "Supported" is also left undefined in
    # build_mark_prompt, while "conveyed" is given an explicit standard.
    #
    # Rate on real data: 5.0% of markings under the GPT judge, 1.1% under Claude.
    # Surfaced per item and aggregated so a precision number cannot be read
    # without it.
    incoherent = bool(covered and claims and precision == 0)
    return {"n_facts": n_facts, "n_covered": len(covered),
            "n_claims": len(claims), "n_unsupported": len(unsupported),
            "incoherent": incoherent,
            "recall": recall, "precision": precision}


def f1(precision, recall):
    """F1 of two AGGREGATE means. None means 'not computable', which is correct
    when combining aggregates but wrong per query: see per_query_f1."""
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def per_query_f1(precision, recall):
    """F1 for a SINGLE query. A query the system scored nothing on is 0.0, not None.

    This distinction is not cosmetic. Reusing f1() here dropped every total-failure
    query from the mean instead of counting it, so the average was taken over only
    the queries a system managed to score on. That flatters precisely the systems
    that fail most often, which is the opposite of what the metric is for: it lifted
    our own system from 0.145 to 0.262 purely by excluding its worst queries.

    None is reserved for genuinely unmeasured (no recall computable at all). A
    summary that produced no checkable claims conveyed nothing, so it scores 0.
    """
    if recall is None:
        return None
    if precision is None or (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mark_system(client, provider, model, system, preds_file, facts_by_qid, split,
                cache_dir=None, quiet=True):
    queries = {q["query_id"]: q["query"] for q in load_queries(split)}
    preds = {json.loads(l)["query_id"]: json.loads(l)["prediction"]
             for l in pathlib.Path(preds_file).read_text(encoding="utf-8").splitlines() if l.strip()}
    items, unparsed = [], 0
    for qid, facts in facts_by_qid.items():
        if qid not in preds:
            continue
        resp = _call(provider, client, model,
                     build_mark_prompt(queries[qid], facts, preds[qid]),
                     MARK_MAX_TOKENS,
                     pathlib.Path(cache_dir) / f"{qid}.json" if cache_dir else None)
        parsed = parse_json_block(resp["text"])
        if parsed is None:
            unparsed += 1; continue
        row = score_marking(parsed, len(facts))
        row["query_id"] = qid
        items.append(row)
    return aggregate(system, model, provider, items, unparsed)


def aggregate(system, model, provider, items, unparsed):
    """Two estimators, because the choice is NOT neutral and one of them is wrong.

    `fact_f1` is the mean of PER-QUERY F1, which is how `eval/scorer.py` averages
    ROUGE (`totals[k] += r[k].fmeasure`, then divided by n). Any comparison of fact
    F1 against ROUGE must use the same estimator or it is not a like-for-like
    comparison.

    `fact_f1_of_means` is F1 computed from the mean precision and mean recall. It
    was the original headline number and it is retained only for audit, because it
    **systematically understates high-variance systems**. Measured spread between the
    two on matched queries: frontier rows move by +0.001 to +0.025, the gold
    reference by +0.001, but our fine-tuned system by **-0.117** and pegasus by
    -0.156. Systems that score zero on many queries and well on others are punished
    by it. Using it against a ROUGE number computed the other way inflated our
    reported gap to the frontier from ~4% to ~160%.
    """
    prec = [i["precision"] for i in items if i["precision"] is not None]
    rec = [i["recall"] for i in items if i["recall"] is not None]
    mp = sum(prec) / len(prec) if prec else None
    mr = sum(rec) / len(rec) if rec else None
    per_query = [per_query_f1(i["precision"], i["recall"]) for i in items]
    per_query = [x for x in per_query if x is not None]
    mf = sum(per_query) / len(per_query) if per_query else None
    # Precision's reliability caveat travels WITH precision, in the same dict, so
    # a row cannot be lifted into a table without it. See score_marking.
    inc = [i for i in items if i.get("incoherent")]
    claims_n = [i["n_claims"] for i in items if i.get("n_claims")]
    return {"system": system, "judge_model": model, "judge_provider": provider,
            "incoherent_markings": len(inc),
            "incoherent_rate": round(len(inc) / len(items), 4) if items else None,
            "mean_claims_denominator": round(sum(claims_n) / len(claims_n), 2) if claims_n else None,
            "precision_caveat": ("precision's denominator is the judge's own claim "
                                 "decomposition, which is not shared across judges "
                                 "(GPT splits frontier summaries 1.2-1.4x finer than "
                                 "Claude); recall's denominator is a cached, shared "
                                 "fact list. Do not read the two as symmetric."),
            "fact_f1_of_means": None if f1(mp, mr) is None else round(f1(mp, mr), 4),
            "judge_vendor": J.vendor_of(model),
            # vendor_of() returns None for our own system name, so this is False
            # for it and True only when a judge marks its own vendor's summaries.
            "self_preference_risk": J.vendor_of(model) == J.vendor_of(system),
            "n_scored": len(items), "unparsed": unparsed,
            "fact_precision": None if mp is None else round(mp, 4),
            "fact_recall": None if mr is None else round(mr, 4),
            "fact_f1": None if mf is None else round(mf, 4),
            "mean_facts_per_query": round(
                sum(i["n_facts"] for i in items) / len(items), 2) if items else None,
            "mean_claims_per_summary": round(
                sum(i["n_claims"] for i in items) / len(items), 2) if items else None,
            "items": items}


GOLD = "GOLD-REFERENCE"

# Published community checkpoints, already scored under ROUGE in results/rows/.
# These are the cheap test of whether the ROUGE-vs-fact divergence is a property of
# FINE-TUNED systems generally or is specific to ours: their predictions already
# exist, so the only cost is marking. distilbart is the strongest prior QMSum
# specialist; led-base is degenerate under vanilla decoding and acts as a low anchor.
COMMUNITY = ["distilbart", "bart-large-cnn", "pegasus", "led-base"]


def community_preds(model, split=P.SPLIT):
    return P.PREDS / f"row-{model}-{split}.jsonl"


def systems_map(split=P.SPLIT, include_gold=False, include_community=False):
    """`include_gold` adds the reference itself as a contestant.

    It is the calibration row: the facts were extracted FROM the reference, so
    marking the reference against them must recover recall near 1.0. Whatever it
    actually scores is this metric's ceiling, and every other row has to be read
    against that rather than against a theoretical 1.0. The pairwise arm only
    revealed itself as broken once the same control was run, so it is not
    optional here.
    """
    m = {P.OURS: P.ours_preds(split)}
    m.update({s: P.theirs_preds(s, split) for s in P.PANEL})
    if include_community:
        m.update({s: community_preds(s, split) for s in COMMUNITY})
    if include_gold:
        from eval.judge_control import build_reference_preds
        m[GOLD] = build_reference_preds(split)
    return m


def run(systems=None, providers=None, split=P.SPLIT, n=None, clients=None, quiet=False,
        extra_systems=None):
    """`extra_systems` maps a label to a preds path, for ad-hoc rows such as a
    decode variant that is not one of the standing systems. Marking is cached per
    (judge, system, query) so re-running an existing row re-bills nothing."""
    providers = providers or P.JUDGE_PROVIDERS
    n = n or protocol.JUDGE_SAMPLE_N
    clients = clients or {}
    all_systems = systems_map(split, include_gold=True, include_community=True)
    all_systems.update({k: pathlib.Path(v) for k, v in (extra_systems or {}).items()})
    systems = systems or [s for s in all_systems if s not in (GOLD, *COMMUNITY)]
    ours_preds = {json.loads(l)["query_id"] for l in
                  P.ours_preds(split).read_text(encoding="utf-8").splitlines() if l.strip()}
    sample = J.pick_sample(ours_preds, n)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = []
    for provider in providers:
        model = J.DEFAULT_JUDGE_MODEL[provider]
        if provider not in clients:
            clients[provider] = P.make_client(provider)
        if not quiet:
            print(f"[facts] extracting reference facts  <-  {model} (n={n})", flush=True)
        facts, failed = extract_facts(clients[provider], provider, model, split, sample,
                                      cache_dir=CACHE_ROOT / f"extract__{model}", quiet=quiet)
        (OUT_DIR / f"{facts_tag(model, split)}.json").write_text(
            json.dumps({"split": split, "n": n, "failed": failed, "facts": facts}, indent=2),
            encoding="utf-8")
        if not quiet:
            total = sum(len(v) for v in facts.values())
            print(f"          {len(facts)} queries, {total} facts "
                  f"({total / len(facts):.1f} per query)" if facts else "          none", flush=True)
        for system in systems:
            if not quiet:
                print(f"[facts] marking {system}  <-  {model}", flush=True)
            r = mark_system(clients[provider], provider, model, system, all_systems[system],
                            facts, split, cache_dir=CACHE_ROOT / f"mark__{model}__{system}",
                            quiet=quiet)
            r["split"] = split
            r["n_requested"] = n
            (OUT_DIR / f"{result_tag(system, model, n, split)}.json").write_text(
                json.dumps(r, indent=2), encoding="utf-8")
            if not quiet:
                print(f"          P {r['fact_precision']}  R {r['fact_recall']}  "
                      f"F1 {r['fact_f1']}  (n={r['n_scored']}, unparsed {r['unparsed']})",
                      flush=True)
            runs.append(r)
    return runs


def result_tag(system, judge_model, n, split=P.SPLIT):
    """Sample size AND split are part of the filename for anything but the defaults.

    A full-split run under the default name silently overwrote an n=60 row and put
    a 281-query number in the same table as five 60-query ones. The 'Scored' column
    showed it, but nothing stopped a reader comparing across sample sizes. Keying on
    n means the two coexist instead of one clobbering the other.

    Split was added 2026-07-29 for the same reason, before it could bite: a val run
    at the default n would have overwritten the test row of the same name, and val
    and test rows are not comparable. Test keeps its bare name so every existing
    artifact keeps working.
    """
    base = f"{system}__judge-{judge_model}"
    if n != protocol.JUDGE_SAMPLE_N:
        base = f"{base}__n{n}"
    return base if split == P.SPLIT else f"{base}__{split}"


def facts_tag(judge_model, split=P.SPLIT, n=None):
    """Filename for a reference-fact decomposition, keyed on BOTH n and split.

    This one already fired. `facts__gpt-5.6-sol.json` carried no n suffix, so the
    n=281 decomposition (2,067 facts) and the n=60 one (474 facts) resolved to the
    same path and the later run silently replaced the earlier. It was found on
    2026-07-29 when a control that should have drawn ~118 reference facts drew 13.
    Nothing had to be re-billed only because the per-query extraction cache is
    keyed on query_id and survived; see `rebuild_facts_from_cache`.

    Measurement rule 11 said sample size belongs in the filename. `result_tag`
    obeyed it and this did not.
    """
    base = f"facts__{judge_model}"
    if n is not None and n != protocol.JUDGE_SAMPLE_N:
        base = f"{base}__n{n}"
    return base if split == P.SPLIT else f"{base}__{split}"


def rebuild_facts_from_cache(judge_model, split=P.SPLIT, quiet=True):
    """Reconstruct a decomposition from the per-query extraction cache.

    The cache is keyed on query_id, so it holds every query ever extracted
    regardless of which run's aggregate file survived. Free: no API calls.
    """
    cache_dir = CACHE_ROOT / f"extract__{judge_model}"
    facts = {}
    for p in sorted(cache_dir.glob("*.json")):
        parsed = parse_json_block(json.loads(p.read_text(encoding="utf-8"))["text"])
        got = parsed.get("facts") if parsed else None
        if isinstance(got, list) and got:
            facts[p.stem] = [str(f) for f in got]
    if not quiet:
        print(f"rebuilt {len(facts)} queries, {sum(len(v) for v in facts.values())} facts")
    return facts


def load_runs(systems=None, providers=None, n=None, split=P.SPLIT):
    systems = systems or list(systems_map())
    providers = providers or P.JUDGE_PROVIDERS
    n = n or protocol.JUDGE_SAMPLE_N
    runs = []
    for system, provider in itertools.product(systems, providers):
        f = OUT_DIR / f"{result_tag(system, J.DEFAULT_JUDGE_MODEL[provider], n, split)}.json"
        if f.exists():
            runs.append(json.loads(f.read_text(encoding="utf-8")))
    return runs


def render_markdown(runs):
    by_judge = {}
    for r in runs:
        by_judge.setdefault(r["judge_model"], []).append(r)
    L = ["### Fact-level judging", "",
         "Reference decomposed into atomic facts by the judge, which never sees a system "
         "summary while doing so; every system is then scored pointwise against the "
         "identical list. Recall = reference facts conveyed. Precision = summary claims "
         "the reference supports. Unlike the pairwise panel there is no A/B ordering, and "
         "unlike ROUGE a paraphrase counts.", ""]
    for model, group in by_judge.items():
        group.sort(key=lambda r: -(r["fact_f1"] or 0))
        sizes = {r["n_scored"] for r in group}
        if len(sizes) > 1:
            L += [f"> **WARNING: mixed sample sizes in this table ({sorted(sizes)}).** "
                  "These rows are NOT comparable to each other. Rebuild with a single n "
                  "before quoting any comparison from it.", ""]
        L += [f"**Judge: {model}**", "",
              "| System | Fact P | Fact R | Fact F1 | Claims/summary | Scored |",
              "|---|---|---|---|---|---|"]
        for r in group:
            star = "**" if r["system"] == P.OURS else ""
            fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
            L.append(f"| {star}{r['system']}{star} | {fmt(r['fact_precision'])} | "
                     f"{fmt(r['fact_recall'])} | {fmt(r['fact_f1'])} | "
                     f"{r['mean_claims_per_summary']} | {r['n_scored']} |")
        L.append("")
    return "\n".join(L)


def main():
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--judges", nargs="*", default=None, choices=P.JUDGE_PROVIDERS)
    ap.add_argument("--n", type=int, default=protocol.JUDGE_SAMPLE_N)
    ap.add_argument("--split", default=P.SPLIT, choices=["val", "test"],
                    help="val for decision-shaped questions; test is for reporting only")
    ap.add_argument("--preds", nargs="*", default=None, metavar="LABEL=PATH",
                    help="ad-hoc rows, e.g. density-min150=results/preds/density-min150-val.jsonl")
    a = ap.parse_args()
    extra = dict(p.split("=", 1) for p in (a.preds or []))
    if a.cmd == "run":
        run(systems=a.systems, providers=a.judges, n=a.n, split=a.split,
            extra_systems=extra)
    runs = load_runs(systems=a.systems, providers=a.judges, n=a.n, split=a.split)
    if not runs:
        print("no fact runs found; run first"); return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = render_markdown(runs)
    (OUT_DIR / "FACTS.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
