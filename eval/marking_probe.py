"""Controlled probe: do fact recall and fact precision use the SAME matching relation?

`build_mark_prompt` asks one judge call for two things:

    (a) which reference facts the summary CONVEYS      -> drives recall
    (b) which summary claims are NOT SUPPORTED by them -> drives precision

If these induced the same claim-to-fact matching, then a summary consisting of
exactly one sentence that matches exactly one fact would be scored either
(covered=[1], unsupported=[]) or (covered=[], unsupported=[1]). Never both, and
never neither: "this sentence conveys the fact" and "this sentence is unsupported
by the fact" would be complementary.

Observed on real data: 3.3% to 6.8% of queries, across every system and both
judges, come back with facts covered AND every claim unsupported. That is only
possible if (a) and (b) apply different standards.

This isolates WHICH difference does it, on synthetic single-fact inputs where the
right answer is not in doubt. Each condition perturbs exactly one property of a
sentence that otherwise states the reference fact verbatim:

    IDENTICAL     the fact, word for word            -> expect covered, supported
    PARAPHRASE    same content, different words      -> expect covered, supported
    ATTRIBUTION   same content, different speaker    -> the suspected culprit
    ADDED_DETAIL  the fact plus extra unstated info  -> the other suspect
    UNRELATED     nothing to do with the fact        -> expect uncovered, unsupported

A condition is INCOHERENT when the judge says the fact is conveyed and also says
the only claim conveying it is unsupported, because those two answers cannot both
inform a meaningful F1.

    python -m eval.marking_probe --judges gpt --repeat 3
"""
import argparse, json, pathlib

from eval import judge as J
from eval import judge_facts as F
from eval import run_judge_panel as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "marking_probe"
CACHE = OUT / "cache"

QUERY = "What did Barry Hughes think about who would build the infrastructure?"
FACT = "Barry Hughes thought that the police might build the infrastructure."

CONDITIONS = {
    "IDENTICAL": "Barry Hughes thought that the police might build the infrastructure.",
    "PARAPHRASE": "Barry Hughes believed the police could construct the infrastructure.",
    "ATTRIBUTION": "Sara Glen thought that the police might build the infrastructure.",
    "ADDED_DETAIL": ("Barry Hughes thought that the police might build the "
                     "infrastructure using their existing commissioning powers."),
    "UNRELATED": "The committee agreed to break for lunch at one o'clock.",
}

# What a single coherent matching relation would produce.
EXPECTED = {
    "IDENTICAL": (True, True),      # (fact covered, claim supported)
    "PARAPHRASE": (True, True),
    "ATTRIBUTION": None,            # either answer is defensible, but not BOTH
    "ADDED_DETAIL": None,
    "UNRELATED": (False, False),
}


def run_condition(client, provider, model, summary, cache_key=None):
    prompt = F.build_mark_prompt(QUERY, [FACT], summary)
    resp = F._call(provider, client, model, prompt, F.MARK_MAX_TOKENS,
                   CACHE / f"{cache_key}.json" if cache_key else None)
    parsed = F.parse_json_block(resp["text"])
    if not parsed:
        return {"parsed": False, "raw": resp["text"]}
    claims = parsed.get("summary_claims") or []
    unsupported = {int(x) for x in parsed.get("unsupported_claim_numbers", [])
                   if str(x).strip().lstrip("-").isdigit()}
    covered = {int(x) for x in parsed.get("covered_fact_numbers", [])
               if str(x).strip().lstrip("-").isdigit()}
    n_supported = len(claims) - len([u for u in unsupported if 1 <= u <= len(claims)])
    fact_covered = 1 in covered
    return {"parsed": True, "n_claims": len(claims), "covered": sorted(covered),
            "unsupported": sorted(unsupported), "fact_covered": fact_covered,
            "any_claim_supported": n_supported > 0,
            # The signature of the defect: the fact is conveyed, yet nothing
            # conveying it counts as supported.
            "incoherent": bool(fact_covered and claims and n_supported == 0),
            "claims": claims}


# --- scaling probe -----------------------------------------------------------
# The single-fact probe above came back coherent on every condition, including
# ATTRIBUTION, which refutes the "the two sub-questions use different standards"
# hypothesis at N=1. The real incoherent cases all have many facts and many
# claims (test_10_s2: 9 claims against 7 facts), and there an attribution
# mismatch WAS credited as covered. So the next variable to isolate is N.
#
# Same perturbation throughout (every claim re-attributed from the named speaker
# to "the team"), only the number of fact/claim pairs changes.

ATTRIBUTES = [
    "the remote control should be hand-sized",
    "the remote control should be light",
    "the remote control should have a call button",
    "the remote control should be water-resistant",
    "the remote control should have a signal",
    "the remote control should be robust",
    "the remote control should have a home",
    "the remote control should be mobile",
    "the remote control should be easy to find",
]
SCALE_QUERY = "What was discussed about the remote control design?"


def scale_inputs(n):
    facts = [f"The Project Manager suggested that {a}." for a in ATTRIBUTES[:n]]
    summary = " ".join(f"The team agreed that {a}." for a in ATTRIBUTES[:n])
    return facts, summary


def run_scale(client, provider, model, n, cache_key=None):
    facts, summary = scale_inputs(n)
    prompt = F.build_mark_prompt(SCALE_QUERY, facts, summary)
    resp = F._call(provider, client, model, prompt, F.MARK_MAX_TOKENS,
                   CACHE / f"{cache_key}.json" if cache_key else None)
    parsed = F.parse_json_block(resp["text"])
    if not parsed:
        return {"parsed": False, "n_facts": n}
    claims = parsed.get("summary_claims") or []
    unsupported = {int(x) for x in parsed.get("unsupported_claim_numbers", [])
                   if str(x).strip().lstrip("-").isdigit()}
    covered = {int(x) for x in parsed.get("covered_fact_numbers", [])
               if str(x).strip().lstrip("-").isdigit()}
    covered = {c for c in covered if 1 <= c <= n}
    n_supported = len(claims) - len([u for u in unsupported if 1 <= u <= len(claims)])
    return {"parsed": True, "n_facts": n, "n_claims": len(claims),
            "n_covered": len(covered), "n_supported": n_supported,
            "recall": len(covered) / n,
            "precision": (n_supported / len(claims)) if claims else None,
            "incoherent": bool(covered and claims and n_supported == 0)}


def run_scaling(providers=("gpt",), sizes=(1, 3, 6, 9), repeat=2, clients=None):
    clients = clients or {}
    rows = []
    for provider in providers:
        model = J.DEFAULT_JUDGE_MODEL[provider]
        if provider not in clients:
            clients[provider] = P.make_client(provider)
        for n in sizes:
            for r in range(repeat):
                res = run_scale(clients[provider], provider, model, n,
                                cache_key=f"scale__{model}__n{n}__{r}")
                res.update({"judge": model, "rep": r})
                rows.append(res)
    return rows


def run(providers=("gpt",), repeat=1, clients=None):
    clients = clients or {}
    rows = []
    for provider in providers:
        model = J.DEFAULT_JUDGE_MODEL[provider]
        if provider not in clients:
            clients[provider] = P.make_client(provider)
        for name, summary in CONDITIONS.items():
            for r in range(repeat):
                res = run_condition(clients[provider], provider, model, summary,
                                    cache_key=f"{model}__{name}__{r}")
                res.update({"condition": name, "judge": model, "rep": r})
                rows.append(res)
    return rows


def summarize(rows):
    by = {}
    for r in rows:
        k = (r["judge"], r["condition"])
        by.setdefault(k, []).append(r)
    out = []
    for (judge, cond), rs in by.items():
        n = len(rs)
        out.append({"judge": judge, "condition": cond, "n": n,
                    "fact_covered": sum(1 for r in rs if r.get("fact_covered")),
                    "claim_supported": sum(1 for r in rs if r.get("any_claim_supported")),
                    "incoherent": sum(1 for r in rs if r.get("incoherent"))})
    return out


def render_markdown(summary):
    L = ["# Marking probe: do precision and recall share one matching relation?", "",
         "One reference fact, one-sentence summaries perturbing exactly one property.",
         "`incoherent` = the judge says the fact is conveyed AND says every claim "
         "conveying it is unsupported.", "",
         "| judge | condition | n | fact covered | claim supported | **incoherent** |",
         "|---|---|---|---|---|---|"]
    for s in sorted(summary, key=lambda x: (x["judge"], x["condition"])):
        L.append(f"| {s['judge']} | {s['condition']} | {s['n']} | {s['fact_covered']}/{s['n']} "
                 f"| {s['claim_supported']}/{s['n']} | **{s['incoherent']}/{s['n']}** |")
    return "\n".join(L) + "\n"


def main():
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="*", default=["gpt"], choices=P.JUDGE_PROVIDERS)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    rows = run(a.judges, a.repeat)
    (OUT / "probe.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    s = summarize(rows)
    md = render_markdown(s)
    (OUT / "PROBE.md").write_text(md, encoding="utf-8")
    print(md)
    for r in rows:
        if r.get("incoherent"):
            print(f"  [{r['judge']} {r['condition']} rep{r['rep']}] covered={r['covered']} "
                  f"unsupported={r['unsupported']} claims={r['claims']}")


if __name__ == "__main__":
    main()
