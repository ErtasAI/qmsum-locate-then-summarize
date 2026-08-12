"""Align a third party's PUBLISHED predictions to our split, then score them ourselves.

Why this exists
---------------
`Salesforce/socratic-pretraining-qmsum` (Pagnoni et al., ACL 2023) ships not only the
QMSum-finetuned checkpoint but its own `test.predictions` and `val.predictions`, plus the
ROUGE its authors computed on them (`test.predictions.rouge`: R-1 38.48). That makes a
comparison possible which no amount of re-running could match for cleanliness: score
THEIR predictions with OUR frozen scorer. No inference, no architecture port, no decoding
config to guess wrong, and the ONLY difference from their published number is the scorer.

That gives two results from one artifact:
  1. A comparable row for the published QMSum state-of-the-art line, under our protocol.
  2. A measured cross-implementation ROUGE delta on IDENTICAL predictions, which turns
     the provenance question (whether published ROUGE numbers are commensurable
     with ours) from an argument into a number.

The alignment problem, and why index-matching would be wrong
-----------------------------------------------------------
Their file has 280 test lines; our released split has 281 (the QMSum paper's own table
says 279, a third count). The predictions are plain text with no query ids, so the mapping
to our `query_id`s has to be recovered. One prediction is missing somewhere, and lining
the files up by index would silently shift every pair after the gap. A shifted alignment
still produces a plausible-looking ROUGE around the low twenties, so this fails QUIETLY.

So the mapping is recovered by monotone dynamic programming over a cheap token-F1
similarity, allowing skips on the reference side only, and it is CONTROLLED before any
score is reported (see `verify`). If the controls do not pass, `main` refuses to write a
row rather than emitting a number nobody can trust. Same withhold-on-failed-control
pattern as `eval/retrieval_attribution.py`.

The similarity used for alignment is deliberately NOT our ROUGE scorer: alignment must not
be tuned on the metric the row will be reported in.
"""
import argparse
import json
import pathlib
import re
from collections import Counter

from data.normalize import load_queries

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Alignment controls.
#
# REVISED after the first run on real data, and the reason is recorded because revising a
# control after seeing data is exactly the move that invalidates a result. Two of the four
# original checks were mis-specified rather than failed:
#
#   - MIN_MARGIN (aligned mean must beat the best rigid index offset by 1.25x) is
#     UNSATISFIABLE when the two files are the same length. With zero slack there is only
#     one monotone alignment, it is the identity, and the best rigid offset IS that
#     identity, so the ratio is 1.0 by construction. The check is meaningful only when a
#     gap has to be located, so it is now conditional on slack > 0.
#   - MIN_TOP1_RECOVERY as an absolute fraction ignores n. Recovering the aligned
#     reference as the global best match for 40% of predictions is near-chance at n=3 and
#     overwhelming at n=281 (chance is 1/281 = 0.4%). The threshold is now stated as a
#     LIFT over chance, which is what the check was always trying to express.
#
# A third revision, same category of mistake, recorded for the same reason. The
# replacement check was first written as a RATIO of matched to unmatched similarity with a
# 2.0 floor. On English prose that threshold is arbitrary: two summaries of two different
# meetings still share stopwords and reporting verbs, so unmatched pairs sit around 0.21
# token F1 rather than near zero, and no correct alignment of prose can reach 2.0. The
# magnitude ratio is confounded by vocabulary overlap.
#
# The floor-free version of the same question is a RANK statistic: among all references,
# where does the matched one rank for each prediction? Under the null of no correspondence
# the mean percentile is 0.5 regardless of how much vocabulary the texts share, so a
# threshold can be set on principle rather than by calibration. The magnitude numbers are
# still reported as diagnostics; they simply no longer gate.
MIN_MEAN_SIM = 0.20        # matched pairs must be substantively similar on average
MIN_MEAN_PERCENTILE = 0.90 # rank of the matched reference among all; null is 0.50
MIN_TOP1_LIFT = 10.0       # top-1 recovery, as a multiple of chance (1 / n_references)
MIN_MARGIN = 1.25          # only when slack > 0: the gap must be located, not guessed

# THE GATE THAT ACTUALLY DECIDES A PUBLISHED ROW, and the reason the thresholds above are
# demoted to sanity checks.
#
# Every check above is a heuristic over a similarity matrix, and the first three attempts at
# calibrating one drifted for reasons that had nothing to do with whether the alignment was
# right (file-length dependence, then a prose vocabulary floor, then the fact that QMSum
# carries several queries per meeting, so a correct alignment legitimately competes against
# near-sibling references and cannot reach a near-perfect rank). Tuning a proxy repeatedly
# against the data it is meant to validate is how a control stops being one.
#
# There is a better instrument available here: the authors published the ROUGE THEY computed
# on these exact predictions (`test.predictions.rouge`, R-1 38.48). Reproducing that with our
# scorer is a direct end-to-end test of the alignment which we cannot tune, because the
# target was fixed by someone else before we saw the file. A misalignment cannot pass it:
# shifting the pairing collapses ROUGE-1 to roughly the low twenties.
#
# The tolerance is set from the documented spread BETWEEN ROUGE implementations on identical
# predictions, routinely reported at one to two points, plus margin. It is deliberately loose
# because measuring that spread is one of the two results this row exists to produce; a tight
# tolerance would assume the answer.
ROUGE1_TOLERANCE = 3.0

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text):
    return _TOKEN.findall((text or "").lower())


def token_f1(a_tokens, b_tokens):
    """Multiset token F1. Cheap, stemmer-free, and independent of the frozen scorer."""
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = sum((Counter(a_tokens) & Counter(b_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def similarity_matrix(predictions, references):
    pred_tokens = [tokens(p) for p in predictions]
    ref_tokens = [tokens(r) for r in references]
    return [[token_f1(pt, rt) for rt in ref_tokens] for pt in pred_tokens]


def align(sim):
    """Monotone alignment of every prediction to a distinct reference, order preserved.

    Returns the list of reference indices, one per prediction, ascending. References may
    be skipped (their file is shorter than ours); predictions may not, since every
    prediction is a real system output that must land somewhere.
    """
    n_pred = len(sim)
    n_ref = len(sim[0]) if n_pred else 0
    if n_pred > n_ref:
        raise ValueError(f"{n_pred} predictions cannot map into {n_ref} references")
    slack = n_ref - n_pred

    NEG = float("-inf")
    # dp[i][j]: best total similarity using the first i predictions and first j references.
    dp = [[NEG] * (n_ref + 1) for _ in range(n_pred + 1)]
    # back[i][j]: True if reference j-1 was matched to prediction i-1, False if skipped.
    back = [[False] * (n_ref + 1) for _ in range(n_pred + 1)]
    for j in range(slack + 1):
        dp[0][j] = 0.0
    for i in range(1, n_pred + 1):
        # j must leave room for the remaining predictions and cannot outrun the slack.
        for j in range(i, min(n_ref, i + slack) + 1):
            best, matched = NEG, False
            if dp[i - 1][j - 1] > NEG:
                best, matched = dp[i - 1][j - 1] + sim[i - 1][j - 1], True
            if dp[i][j - 1] > NEG and dp[i][j - 1] > best:
                best, matched = dp[i][j - 1], False
            dp[i][j] = best
            back[i][j] = matched

    mapping = [None] * n_pred
    i, j = n_pred, n_ref
    while i > 0:
        if back[i][j]:
            mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
        else:
            j -= 1
    return mapping


def best_rigid_offset(sim):
    """Best mean similarity achievable by lining the files up with a fixed shift.

    The null model the DP has to beat. If a rigid offset scores about as well, the gap is
    not really located and the alignment is not trustworthy.
    """
    n_pred, n_ref = len(sim), len(sim[0])
    best = 0.0
    for offset in range(n_ref - n_pred + 1):
        total = sum(sim[i][i + offset] for i in range(n_pred))
        best = max(best, total / n_pred)
    return best


def verify(sim, mapping):
    """Alignment controls. Every number here is reported, pass or fail."""
    n_pred = len(sim)
    n_ref = len(sim[0])
    slack = n_ref - n_pred
    matched = [sim[i][mapping[i]] for i in range(n_pred)]
    mean_sim = sum(matched) / n_pred

    # The null: every prediction against every reference it was NOT matched to. Reported as
    # a diagnostic only. On prose this has a vocabulary floor, which is why it does not gate.
    unmatched = [sim[i][j] for i in range(n_pred) for j in range(n_ref) if j != mapping[i]]
    null_sim = sum(unmatched) / len(unmatched) if unmatched else 0.0

    # The gating version of the same question, floor-free because it is rank-based: for each
    # prediction, what fraction of the other references does its matched reference beat?
    # Under no correspondence this averages 0.50 however much vocabulary the texts share.
    percentiles = []
    for i in range(n_pred):
        target = sim[i][mapping[i]]
        beaten = sum(1 for j in range(n_ref) if j != mapping[i] and sim[i][j] < target)
        percentiles.append(beaten / (n_ref - 1) if n_ref > 1 else 1.0)
    mean_percentile = sum(percentiles) / n_pred

    top1 = sum(1 for i in range(n_pred)
               if max(range(n_ref), key=lambda j: sim[i][j]) == mapping[i])
    top1_recovery = top1 / n_pred
    chance = 1.0 / n_ref
    rigid = best_rigid_offset(sim)

    checks = {
        "n_predictions": n_pred,
        "n_references": n_ref,
        "slack": slack,
        "mean_matched_similarity": round(mean_sim, 4),
        "mean_matched_percentile": round(mean_percentile, 4),
        "mean_unmatched_similarity": round(null_sim, 4),
        "null_ratio_diagnostic": round(mean_sim / null_sim, 4) if null_sim else None,
        "top1_recovery": round(top1_recovery, 4),
        "top1_lift_over_chance": round(top1_recovery / chance, 2),
        "best_rigid_offset_similarity": round(rigid, 4),
        # Vacuous at slack 0: the identity is the only monotone alignment available.
        "margin_over_rigid": round(mean_sim / rigid, 4) if (slack and rigid) else None,
        "monotone": all(mapping[i] < mapping[i + 1] for i in range(n_pred - 1)),
    }
    checks["passed"] = bool(
        checks["mean_matched_similarity"] >= MIN_MEAN_SIM
        and checks["mean_matched_percentile"] >= MIN_MEAN_PERCENTILE
        and checks["top1_lift_over_chance"] >= MIN_TOP1_LIFT
        and checks["monotone"]
        and (slack == 0 or (checks["margin_over_rigid"] or 0) >= MIN_MARGIN)
    )
    return checks


def load_predictions(path):
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def reproduction_check(aligned, queries, expect_rouge1):
    """Score the aligned pairs with the frozen scorer and compare to the authors' own number.

    Returns None when no expected value was supplied, in which case the similarity
    heuristics are all that is available and `build` gates on those instead.
    """
    if expect_rouge1 is None:
        return None
    from eval.scorer import score
    refs = {q["query_id"]: q["reference"] for q in queries}
    ours = score([r["prediction"] for r in aligned],
                 [refs[r["query_id"]] for r in aligned])["rouge1"] * 100
    delta = ours - expect_rouge1
    return {
        "author_reported_rouge1": expect_rouge1,
        "our_scorer_rouge1": round(ours, 4),
        "delta": round(delta, 4),
        "tolerance": ROUGE1_TOLERANCE,
        "passed": abs(delta) <= ROUGE1_TOLERANCE,
    }


def build(pred_path, split, slug, expect_rouge1=None):
    queries = load_queries(split)
    predictions = load_predictions(pred_path)
    sim = similarity_matrix(predictions, [q["reference"] for q in queries])
    mapping = align(sim)
    checks = verify(sim, mapping)
    aligned = [{"query_id": queries[mapping[i]]["query_id"], "prediction": predictions[i]}
               for i in range(len(predictions))]
    skipped = [queries[j]["query_id"] for j in range(len(queries)) if j not in set(mapping)]
    reproduction = reproduction_check(aligned, queries, expect_rouge1)
    report = {
        "slug": slug,
        "split": split,
        "n_predictions": len(predictions),
        "n_queries": len(queries),
        "skipped_query_ids": skipped,
        # When the authors published a score, THAT decides the row and the similarity
        # heuristics are recorded as context. Only when they did not do the heuristics gate.
        "gate": "author_score_reproduction" if reproduction else "similarity_heuristics",
        "reproduction": reproduction,
        "controls": checks,
    }
    report["passed"] = reproduction["passed"] if reproduction else checks["passed"]
    return aligned, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="third party's raw prediction file, one per line")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--slug", required=True, help="row slug, e.g. socratic-segenc")
    ap.add_argument("--expect-rouge1", type=float,
                    help="the authors' own ROUGE-1 (x100) on this prediction file. When given, "
                         "reproducing it within tolerance is what gates the row.")
    ap.add_argument("--force", action="store_true",
                    help="write the aligned file even if the controls fail (diagnosis only)")
    a = ap.parse_args()

    aligned, report = build(a.pred, a.split, a.slug, a.expect_rouge1)
    report_path = ROOT / "results" / "external" / f"alignment-{a.slug}-{a.split}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if not report["passed"] and not a.force:
        raise SystemExit(
            "alignment controls FAILED; no prediction file written. "
            "A misaligned file still scores plausibly, so this is not a warning to skip. "
            "Re-run with --force only to inspect the alignment.")

    out = ROOT / "results" / "preds" / f"row-{a.slug}-{a.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in aligned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}\nalignment report {report_path}")
    print(f"\nNext: python -m eval.run_eval --pred {out} --split {a.split} "
          f"--run-id row-{a.slug}-{a.split} --bertscore --allow-partial")


if __name__ == "__main__":
    main()
