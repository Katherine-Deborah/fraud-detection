"""End-to-end Session 7 demo: registers all 5 Session 5/6 candidate models,
bootstraps the first production model, walks Random Forest through all 4
stages (Staging -> Shadow -> Canary -> Production), walks Isolation Forest
in and shows it rejected at the cheapest gate, and demonstrates the manual
rollback path (and its reversal). Every decision is printed and logged to
the model-registry-audit MLflow experiment. See docs/model_registry.md for
the narrative writeup of a real run of this script.

Usage (from .venv-registry):
    python training/run_registry_walkthrough.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import canary_harness, promotion_criteria, registry, shadow_harness  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    client = registry.get_client()

    section("1. Register all 5 candidate model versions (all start in 'staging')")
    versions = {}
    for run_name in ["logistic_regression", "random_forest", "isolation_forest", "lstm", "random_forest_sagemaker"]:
        v = registry.register_candidate(client, run_name)
        versions[run_name] = v
        print(f"registered v{v.version}: {run_name} (run {v.run_id})")

    section("2. Bootstrap: promote logistic_regression straight to production (no predecessor to compare against)")
    registry.promote(
        client, versions["logistic_regression"], registry.STAGE_PRODUCTION,
        reason="bootstrap: first model ever registered, no production model exists yet to gate against",
    )

    section("3. Candidate: random_forest -- Staging -> Shadow gate")
    rf = versions["random_forest"]
    production = registry.get_current_production(client)
    rf_auc_pr = registry.source_run_metrics(client, rf)["auc_pr"]
    prod_auc_pr = registry.source_run_metrics(client, production)["auc_pr"]
    passed, detail = promotion_criteria.gate_staging_to_shadow(rf_auc_pr, prod_auc_pr)
    print(f"gate_staging_to_shadow: passed={passed} detail={detail}")
    reason = f"offline AUC-PR {rf_auc_pr:.4f} vs production {prod_auc_pr:.4f} (ratio {detail['ratio']:.2f})"
    if not passed:
        registry.reject(client, rf, reason, detail)
        raise SystemExit("random_forest failed the staging->shadow gate -- unexpected, stopping demo")
    registry.promote(client, rf, registry.STAGE_SHADOW, reason, detail)

    section("4. Candidate: random_forest -- Shadow harness (scores shadow-batch, predictions not acted on)")
    shadow_passed, shadow_detail = shadow_harness.run_shadow(client, rf)
    print(f"shadow gate: passed={shadow_passed}")
    reason = (
        f"shadow batch ({shadow_detail['n_rows']:,} rows): candidate recall={shadow_detail['candidate_recall']:.4f} "
        f"fpr={shadow_detail['candidate_fpr']:.4f} vs production recall={shadow_detail['production_recall']:.4f}"
    )
    if not shadow_passed:
        registry.reject(client, rf, reason, shadow_detail)
        raise SystemExit("random_forest failed the shadow->canary gate -- unexpected, stopping demo")
    registry.promote(client, rf, registry.STAGE_CANARY, reason, shadow_detail)

    section("5. Candidate: random_forest -- Canary harness (10% of accounts, predictions acted on)")
    canary_passed, canary_detail = canary_harness.run_canary(client, rf)
    reason = (
        f"canary slice ({canary_detail['n_canary_rows']:,} rows, {canary_detail['canary_fraud_count']} fraud): "
        f"recall={canary_detail['canary_recall']:.4f} fpr={canary_detail['canary_fpr']:.4f} "
        f"vs production recall={canary_detail['production_recall']:.4f}"
    )
    if not canary_passed:
        registry.reject(client, rf, reason, canary_detail)
        raise SystemExit("random_forest failed the canary->production gate -- unexpected, stopping demo")
    registry.promote(client, rf, registry.STAGE_PRODUCTION, reason, canary_detail)
    registry.set_stage(client, production.version, registry.STAGE_ARCHIVED)
    registry.log_audit_event(
        client, "archive", production, registry.STAGE_PRODUCTION, registry.STAGE_ARCHIVED,
        True, f"superseded by v{rf.version} (random_forest)",
    )
    print(f"random_forest (v{rf.version}) is now production; logistic_regression (v{production.version}) archived")

    section("6. Candidate: isolation_forest -- Staging -> Shadow gate (expected to fail)")
    iso = versions["isolation_forest"]
    current_production = registry.get_current_production(client)
    iso_auc_pr = registry.source_run_metrics(client, iso)["auc_pr"]
    current_prod_auc_pr = registry.source_run_metrics(client, current_production)["auc_pr"]
    passed, detail = promotion_criteria.gate_staging_to_shadow(iso_auc_pr, current_prod_auc_pr)
    print(f"gate_staging_to_shadow: passed={passed} detail={detail}")
    reason = f"offline AUC-PR {iso_auc_pr:.4f} vs production {current_prod_auc_pr:.4f} (ratio {detail['ratio']:.2f})"
    if passed:
        raise SystemExit("isolation_forest unexpectedly passed the staging->shadow gate -- check gate thresholds")
    registry.reject(client, iso, reason, detail)
    print("isolation_forest rejected before any shadow/canary compute was spent on it -- the point of a cheap early gate")

    section("7. lstm and random_forest_sagemaker: registered as additional candidates, left in staging")
    print(
        "Not walked through this session's live demo (see docs/model_registry.md for why: the LSTM's "
        "pyfunc interface expects per-account sequence windows, not the flat feature rows the sklearn "
        "shadow/canary harnesses score, and the SageMaker RF model was trained under a scikit-learn "
        "version this venv can't unpickle -- see requirements-registry.txt). Both remain valid future "
        "candidates sitting in 'staging'."
    )

    section("8. Rollback demo: simulate a post-promotion incident, revert production to logistic_regression")
    registry.rollback(
        client, versions["logistic_regression"],
        reason="simulated production incident discovered after promotion (demo of the rollback mechanism)",
    )
    print(f"production is now v{registry.get_current_production(client).version} "
          f"({registry.get_current_production(client).tags.get(registry.TAG_ALGORITHM)})")

    section("9. Revert the demo rollback: random_forest is genuinely the better model, restore it to production")
    registry.rollback(
        client, rf,
        reason="rollback mechanism demonstrated in step 8; restoring random_forest as the intended final state "
               "(higher AUC-PR, matches docs/model_comparison.md's recommendation)",
    )

    section("Final registry state")
    registry.print_registry_state(client)


if __name__ == "__main__":
    main()
