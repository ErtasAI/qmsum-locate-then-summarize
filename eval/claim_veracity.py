"""Are our 'unsupported' claims false, or merely absent from the reference?

Fact precision marks a claim unsupported when it is not covered by the reference.
The consequence: a claim that is TRUE of the
transcript but that the reference's author did not choose to include is scored
identically to a hallucination. If most of our unsupported claims are true, then
"our claims are wrong" is the wrong reading and precision 0.183 is measuring
reference-mimicry rather than correctness.

This builds blinded packets so that question can be answered against the
TRANSCRIPT instead of against the reference, and aggregates the verdicts.

DESIGN, mirroring the discipline the judge work established:

1. **Blind.** Claims from every source are pooled, shuffled under the frozen seed
   and given opaque ids. A verifier cannot tell our claim from a frontier one.
2. **Controlled.** The gold reference's OWN claims are entered as a hidden
   contestant. They are by construction true of the transcript, so a verifier that
   marks them absent is broken and its verdicts on everything else are void. This
   is the same control that exposed reference-free pairwise judging.
3. **Comparative.** luna's unsupported claims are verified alongside ours. luna
   emits 14.14 claims per summary to our 6.20, so it has far MORE unsupported
   claims in absolute terms. The question is not whether unsupported-but-true
   happens, it is whether it happens at a different RATE for us than for a system
   we are being compared against.

The verification itself is done by agents reading the meeting transcript. Only
packet-building and aggregation live here.

    python -m eval.claim_veracity build --n-queries 16
    python -m eval.claim_veracity aggregate --verdicts <dir>
"""
import argparse, json, pathlib, random

from data.normalize import load_meetings, load_queries
from eval import protocol

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "judge_facts" / "cache"
OUT = ROOT / "results" / "claim_veracity"

JUDGE = "gpt-5.6-sol"
OURS = "m3-fusion-lfm2.5-1.2b"
BASELINE = "gpt-5.6-luna"
GOLD = "GOLD-REFERENCE"

LABELS = ("SUPPORTED_FULL", "SUPPORTED_PARTIAL", "CONTRADICTED", "NOT_IN_TRANSCRIPT")
# Labels that mean "the claim is not false of the transcript".
TRUE_LABELS = ("SUPPORTED_FULL", "SUPPORTED_PARTIAL")
# Below this share of gold claims verified true, the verifier is not usable.
GOLD_CONTROL_FLOOR = 0.85


def load_marking(system, query_id, judge=JUDGE):
    p = CACHE / f"mark__{judge}__{system}" / f"{query_id}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))["text"]
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


def unsupported_claims(system, query_id):
    """Claims this system emitted that the reference did NOT cover.

    `unsupported_claim_numbers` is 1-indexed in the judge's output.
    """
    m = load_marking(system, query_id)
    if not m:
        return []
    claims = m.get("summary_claims") or []
    idx = m.get("unsupported_claim_numbers") or []
    return [claims[i - 1] for i in idx if 1 <= i <= len(claims)]


def all_claims(system, query_id):
    m = load_marking(system, query_id)
    return (m.get("summary_claims") or []) if m else []


def reference_facts(query_id, judge=JUDGE, split="test"):
    """The reference's own atomic facts, used as the control claims.

    Preferred over the GOLD-REFERENCE marking cache, which only covers the n=60
    sample; this decomposition covers all 281 and so gives a control of real size
    on any query selection. These facts were extracted FROM the reference, which
    is a faithful summary of the transcript, so a verifier that cannot confirm
    them against that transcript is not measuring what it claims to.
    """
    fdir = ROOT / "results" / "judge_facts"
    # Prefer the widest decomposition available. The unsuffixed name held only
    # n=60 after a filename collision, which silently shrank this control to 13
    # claims the first time it was built.
    for p in (fdir / f"facts__{judge}__n281.json", fdir / f"facts__{judge}.json"):
        if p.exists():
            facts = json.loads(p.read_text(encoding="utf-8")).get("facts", {})
            if query_id in facts:
                return facts[query_id]
    return []


def build_packets(query_ids, split="test", seed=protocol.SEED):
    """One packet per query: the transcript, plus every claim to verify, blinded."""
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
    queries = {q["query_id"]: q for q in load_queries(split)}
    rng = random.Random(seed)
    packets, key = [], {}
    for qid in query_ids:
        q = queries[qid]
        pool = ([(OURS, c) for c in unsupported_claims(OURS, qid)]
                + [(BASELINE, c) for c in unsupported_claims(BASELINE, qid)]
                + [(GOLD, c) for c in reference_facts(qid)])
        if not pool:
            continue
        rng.shuffle(pool)
        items = []
        for i, (src, claim) in enumerate(pool, 1):
            cid = f"{qid}__c{i:02d}"
            key[cid] = {"source": src, "claim": claim, "query_id": qid}
            items.append({"claim_id": cid, "claim": claim})
        transcript = "\n".join(f"{u['speaker']}: {u['text']}" for u in meetings[q["meeting_id"]])
        packets.append({"query_id": qid, "query": q["query"],
                        "meeting_id": q["meeting_id"], "transcript": transcript,
                        "claims": items})
    return packets, key


def aggregate(key, verdicts):
    """Rate of claims judged true of the transcript, per source.

    Returns None for every rate if the gold control fails, because in that case
    the verifier could not confirm content the transcript definitionally contains
    and no other row it produced means anything.
    """
    by_source = {}
    for cid, meta in key.items():
        v = verdicts.get(cid)
        if v is None:
            continue
        by_source.setdefault(meta["source"], []).append(v)

    rows = {}
    for src, labels in by_source.items():
        n = len(labels)
        counts = {l: labels.count(l) for l in LABELS}
        unknown = n - sum(counts.values())
        rows[src] = {"n": n, **counts, "unlabelled": unknown,
                     "true_rate": round(sum(counts[l] for l in TRUE_LABELS) / n, 4) if n else None}

    gold = rows.get(GOLD)
    control_passed = bool(gold and gold["true_rate"] is not None
                          and gold["true_rate"] >= GOLD_CONTROL_FLOOR)
    return {"rows": rows, "gold_control": gold,
            "gold_control_floor": GOLD_CONTROL_FLOOR,
            "control_passed": control_passed,
            "verdict": ("usable" if control_passed else
                        "VOID: the verifier could not confirm the reference's own "
                        "claims against the transcript, so no row here is meaningful")}


def render_markdown(res):
    L = ["# Are 'unsupported' claims actually false?", "",
         "Fact precision marks a claim unsupported when the REFERENCE does not cover "
         "it. This re-checks those claims against the TRANSCRIPT.", "",
         "| source | n | full | partial | contradicted | not in transcript | **true rate** |",
         "|---|---|---|---|---|---|---|"]
    for src, r in res["rows"].items():
        L.append(f"| {src} | {r['n']} | {r['SUPPORTED_FULL']} | {r['SUPPORTED_PARTIAL']} "
                 f"| {r['CONTRADICTED']} | {r['NOT_IN_TRANSCRIPT']} | **{r['true_rate']}** |")
    L += ["", f"**Control:** gold-reference claims verified true at "
              f"{res['gold_control']['true_rate'] if res['gold_control'] else 'n/a'} "
              f"(floor {res['gold_control_floor']}). **{res['verdict']}**", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "aggregate"])
    ap.add_argument("--queries", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verdicts", help="JSON mapping claim_id -> label")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build":
        packets, key = build_packets(a.queries)
        outdir = pathlib.Path(a.out or (OUT / "packets"))
        outdir.mkdir(parents=True, exist_ok=True)
        for p in packets:
            (outdir / f"{p['query_id']}.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
        (OUT / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
        print(f"{len(packets)} packets -> {outdir}; {len(key)} claims; key -> {OUT/'key.json'}")
    else:
        key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
        verdicts = json.loads(pathlib.Path(a.verdicts).read_text(encoding="utf-8"))
        res = aggregate(key, verdicts)
        (OUT / "veracity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        md = render_markdown(res)
        (OUT / "VERACITY.md").write_text(md, encoding="utf-8")
        print(md)


if __name__ == "__main__":
    main()
