"""Promotion gate functions for the 4-stage rollout (Staging -> Shadow ->
Canary -> Production). Each gate is a pure function: given the metrics it
needs, it returns (passed: bool, detail: dict) -- detail always contains
the numbers the decision was based on, so it can be logged verbatim to the
audit trail (training/registry.py::log_audit_event) without the caller
having to reconstruct anything.

Full rationale for each threshold lives in docs/model_registry.md; this
module is deliberately just the numbers + comparisons, not the essay.
"""

from __future__ import annotations

FPR_TARGET = 0.02  # PROJECT.md #6: recall >= 90% at FPR <= 2%

# Staging -> Shadow: cheap offline regression gate, using each model's
# already-computed *test-set* AUC-PR from its own training run (Session 5/6
# metrics, no new scoring needed). A tolerance below 100% (not "must beat
# production outright") lets a near-parity candidate through -- e.g. a
# simpler/cheaper model that's very slightly behind on ranking quality --
# while still cheaply blocking a candidate that's clearly worse before
# paying for a shadow-scoring pass.
STAGING_TO_SHADOW_AUC_PR_RATIO = 0.90

# Shadow -> Canary: candidate must match production's recall (at each
# model's own pre-selected threshold, see docs/model_registry.md's
# "why thresholds aren't re-picked" note) within a small tolerance on the
# *shared* shadow batch, and must independently respect the FPR<=2% budget
# -- a model that beats production on recall by flagging everything isn't
# a real win.
SHADOW_TO_CANARY_RECALL_TOLERANCE = 0.02  # 2 percentage points
SHADOW_TO_CANARY_FPR_BUDGET = FPR_TARGET

# Canary -> Production: same recall/FPR comparison as the shadow gate, but
# measured on the actual 10% canary traffic slice the candidate is acting
# on for real, with a slightly looser FPR budget (a smaller sample is
# noisier) and a minimum sample-size floor so a lucky/unlucky small slice
# can't singlehandedly promote or sink a candidate.
CANARY_TO_PRODUCTION_RECALL_TOLERANCE = 0.02
CANARY_TO_PRODUCTION_FPR_BUDGET = FPR_TARGET + 0.005
CANARY_MIN_FRAUD_COUNT = 20


def gate_staging_to_shadow(candidate_auc_pr: float, production_auc_pr: float | None) -> tuple[bool, dict]:
    if production_auc_pr is None:
        return True, {"reason": "bootstrap: no production model to compare against yet"}
    ratio = candidate_auc_pr / production_auc_pr if production_auc_pr else 0.0
    passed = ratio >= STAGING_TO_SHADOW_AUC_PR_RATIO
    return passed, {
        "candidate_auc_pr": candidate_auc_pr,
        "production_auc_pr": production_auc_pr,
        "ratio": ratio,
        "required_ratio": STAGING_TO_SHADOW_AUC_PR_RATIO,
    }


def gate_shadow_to_canary(
    candidate_recall: float, candidate_fpr: float, production_recall: float
) -> tuple[bool, dict]:
    fpr_ok = candidate_fpr <= SHADOW_TO_CANARY_FPR_BUDGET
    recall_ok = candidate_recall >= production_recall - SHADOW_TO_CANARY_RECALL_TOLERANCE
    return fpr_ok and recall_ok, {
        "candidate_recall": candidate_recall,
        "candidate_fpr": candidate_fpr,
        "production_recall": production_recall,
        "fpr_budget": SHADOW_TO_CANARY_FPR_BUDGET,
        "recall_tolerance": SHADOW_TO_CANARY_RECALL_TOLERANCE,
        "fpr_ok": fpr_ok,
        "recall_ok": recall_ok,
    }


def gate_canary_to_production(
    canary_recall: float,
    canary_fpr: float,
    production_recall: float,
    canary_fraud_count: int,
) -> tuple[bool, dict]:
    sample_ok = canary_fraud_count >= CANARY_MIN_FRAUD_COUNT
    fpr_ok = canary_fpr <= CANARY_TO_PRODUCTION_FPR_BUDGET
    recall_ok = canary_recall >= production_recall - CANARY_TO_PRODUCTION_RECALL_TOLERANCE
    passed = sample_ok and fpr_ok and recall_ok
    return passed, {
        "canary_recall": canary_recall,
        "canary_fpr": canary_fpr,
        "production_recall": production_recall,
        "canary_fraud_count": canary_fraud_count,
        "min_fraud_count": CANARY_MIN_FRAUD_COUNT,
        "fpr_budget": CANARY_TO_PRODUCTION_FPR_BUDGET,
        "recall_tolerance": CANARY_TO_PRODUCTION_RECALL_TOLERANCE,
        "sample_ok": sample_ok,
        "fpr_ok": fpr_ok,
        "recall_ok": recall_ok,
    }
