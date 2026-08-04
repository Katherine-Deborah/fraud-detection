"""Core MLflow Model Registry operations for the Session 7 staged rollout.

All 5 candidate models trained so far (Sessions 5-6: logistic_regression,
random_forest, isolation_forest, lstm, random_forest_sagemaker) are
registered as **versions of a single registered model**,
`fraud-detection-classifier` -- they are all candidates competing for the
same production slot, not separate products.

Stage vocabulary: MLflow's built-in model version "stage" (None / Staging /
Production / Archived) is both deprecated upstream (in favor of aliases/
tags) and doesn't have room for this project's "Shadow" and "Canary"
stages anyway, so this module deliberately does NOT call
`transition_model_version_stage`. Instead the current stage is the single
source of truth stored in a **model version tag**, `lifecycle_stage`, with
6 possible values:

    staging -> shadow -> canary -> production
                                       |
                                    archived   (superseded by a later
                                                promotion, or manually
                                                rolled back into)
    staging -> rejected  (failed a gate before reaching production)

See docs/model_registry.md for the full promotion criteria between each
stage and the rollback path.

Every stage transition is also logged as its own MLflow run in a separate
"model-registry-audit" experiment (params/metrics/tags describing the
decision) -- this is the audit trail: `lifecycle_stage` tags tell you where
a version is *now*, the audit experiment tells you *how it got there*.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

MLFLOW_TRACKING_URI = "http://localhost:5000"
SOURCE_EXPERIMENT_NAME = "fraud-detection"
AUDIT_EXPERIMENT_NAME = "model-registry-audit"
REGISTERED_MODEL_NAME = "fraud-detection-classifier"

TAG_STAGE = "lifecycle_stage"
TAG_ALGORITHM = "algorithm"
TAG_SOURCE_RUN_ID = "source_run_id"
TAG_TRAINED_VIA = "trained_via"

STAGE_STAGING = "staging"
STAGE_SHADOW = "shadow"
STAGE_CANARY = "canary"
STAGE_PRODUCTION = "production"
STAGE_ARCHIVED = "archived"
STAGE_REJECTED = "rejected"

ALL_STAGES = (
    STAGE_STAGING,
    STAGE_SHADOW,
    STAGE_CANARY,
    STAGE_PRODUCTION,
    STAGE_ARCHIVED,
    STAGE_REJECTED,
)


def get_client(tracking_uri: str = MLFLOW_TRACKING_URI) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


@dataclass
class SourceRun:
    run_id: str
    run_name: str
    metrics: dict[str, float]
    tags: dict[str, str]


def find_latest_run(
    client: MlflowClient, run_name: str, experiment_name: str = SOURCE_EXPERIMENT_NAME
) -> SourceRun:
    """Latest run matching `run_name` in the source training experiment --
    same "most recent wins" convention as training/compare_models.py, which
    matters here because a few run_names (logistic_regression,
    random_forest_sagemaker) have more than one logged run in this
    project's history (a bad early attempt, or a re-registration during
    debugging) and the fixed/final one is always the most recent."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"No MLflow experiment named {experiment_name!r}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"No run named {run_name!r} in experiment {experiment_name!r}")
    run = runs[0]
    return SourceRun(run_id=run.info.run_id, run_name=run_name, metrics=dict(run.data.metrics), tags=dict(run.data.tags))


def _ensure_registered_model(client: MlflowClient, name: str = REGISTERED_MODEL_NAME) -> None:
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(
            name,
            description=(
                "Fraud classifier candidates (Sessions 5-6: logistic_regression, "
                "random_forest, isolation_forest, lstm, random_forest_sagemaker). "
                "One version per training run; current lifecycle stage lives in "
                "each version's 'lifecycle_stage' tag -- see docs/model_registry.md."
            ),
        )


def register_candidate(
    client: MlflowClient, run_name: str, experiment_name: str = SOURCE_EXPERIMENT_NAME
) -> ModelVersion:
    """Create a new registered model version pointing at run_name's logged
    model artifact, tagged lifecycle_stage=staging. Registration only needs
    the artifact's run-relative URI, not the deserialized model object, so
    this works identically for the LSTM (pytorch flavor) and the SageMaker
    RF model (a different, incompatible scikit-learn pickle version, see
    requirements-registry.txt) -- neither ever gets loaded by this
    function."""
    _ensure_registered_model(client)
    source_run = find_latest_run(client, run_name, experiment_name)
    model_uri = f"runs:/{source_run.run_id}/model"

    version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=model_uri,
        run_id=source_run.run_id,
    )
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, TAG_STAGE, STAGE_STAGING)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, TAG_ALGORITHM, run_name)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, TAG_SOURCE_RUN_ID, source_run.run_id)
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version.version, TAG_TRAINED_VIA, source_run.tags.get(TAG_TRAINED_VIA, "local")
    )
    return client.get_model_version(REGISTERED_MODEL_NAME, version.version)


def set_stage(client: MlflowClient, version: str, stage: str) -> None:
    if stage not in ALL_STAGES:
        raise ValueError(f"unknown stage {stage!r}, must be one of {ALL_STAGES}")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, TAG_STAGE, stage)


def get_stage(client: MlflowClient, version: str) -> str:
    mv = client.get_model_version(REGISTERED_MODEL_NAME, version)
    return mv.tags.get(TAG_STAGE, STAGE_STAGING)


def versions_in_stage(client: MlflowClient, stage: str) -> list[ModelVersion]:
    return [
        mv
        for mv in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if mv.tags.get(TAG_STAGE) == stage
    ]


def get_current_production(client: MlflowClient) -> ModelVersion | None:
    versions = versions_in_stage(client, STAGE_PRODUCTION)
    if len(versions) > 1:
        raise RuntimeError(
            f"invariant violated: {len(versions)} versions tagged production "
            f"simultaneously ({[v.version for v in versions]}) -- exactly one is expected"
        )
    return versions[0] if versions else None


def source_run_metrics(client: MlflowClient, version: ModelVersion) -> dict[str, float]:
    run = client.get_run(version.run_id)
    return dict(run.data.metrics)


def log_audit_event(
    client: MlflowClient,
    action: str,
    version: ModelVersion,
    from_stage: str,
    to_stage: str,
    approved: bool,
    reason: str,
    decision_metrics: dict[str, float] | None = None,
) -> str:
    """Log one promotion/rejection/rollback decision as its own run in the
    model-registry-audit experiment -- the human-readable "why" behind
    every lifecycle_stage tag change. Returns the audit run's run_id."""
    mlflow.set_experiment(AUDIT_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{action}_{version.tags.get(TAG_ALGORITHM, '?')}_v{version.version}") as run:
        mlflow.set_tags(
            {
                "action": action,
                "model_version": version.version,
                "algorithm": version.tags.get(TAG_ALGORITHM, "?"),
                "from_stage": from_stage,
                "to_stage": to_stage,
                "approved": str(approved),
                "reason": reason,
            }
        )
        if decision_metrics:
            mlflow.log_metrics(decision_metrics)
        print(f"[audit] {action}: v{version.version} ({version.tags.get(TAG_ALGORITHM)}) "
              f"{from_stage} -> {to_stage} | approved={approved} | {reason}")
        return run.info.run_id


def promote(
    client: MlflowClient,
    version: ModelVersion,
    to_stage: str,
    reason: str,
    decision_metrics: dict[str, float] | None = None,
) -> None:
    from_stage = get_stage(client, version.version)
    set_stage(client, version.version, to_stage)
    log_audit_event(client, "promote", version, from_stage, to_stage, True, reason, decision_metrics)


def reject(
    client: MlflowClient,
    version: ModelVersion,
    reason: str,
    decision_metrics: dict[str, float] | None = None,
) -> None:
    from_stage = get_stage(client, version.version)
    set_stage(client, version.version, STAGE_REJECTED)
    log_audit_event(client, "reject", version, from_stage, STAGE_REJECTED, False, reason, decision_metrics)


def rollback(client: MlflowClient, target_version: ModelVersion, reason: str) -> None:
    """Manual/emergency rollback: `target_version` (must currently be
    `archived`) becomes the new `production`; whatever is currently
    `production` (if anything) is demoted to `archived`. This is the
    generic mechanism -- it doesn't require a canary to have just failed,
    it's callable any time an operator decides the current production
    model needs to be reverted (e.g. a problem noticed after the fact)."""
    if get_stage(client, target_version.version) != STAGE_ARCHIVED:
        raise ValueError(
            f"rollback target v{target_version.version} must be in stage "
            f"'{STAGE_ARCHIVED}', is currently {get_stage(client, target_version.version)!r}"
        )
    current_prod = get_current_production(client)

    set_stage(client, target_version.version, STAGE_PRODUCTION)
    log_audit_event(client, "rollback", target_version, STAGE_ARCHIVED, STAGE_PRODUCTION, True, reason)

    if current_prod is not None:
        set_stage(client, current_prod.version, STAGE_ARCHIVED)
        log_audit_event(client, "rollback", current_prod, STAGE_PRODUCTION, STAGE_ARCHIVED, True, reason)


def print_registry_state(client: MlflowClient) -> None:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    versions = sorted(versions, key=lambda v: int(v.version))
    print(f"\n{'version':>7}  {'algorithm':<24}{'stage':<12}{'trained_via':<12}run_id")
    for v in versions:
        print(
            f"{v.version:>7}  {v.tags.get(TAG_ALGORITHM, '?'):<24}"
            f"{v.tags.get(TAG_STAGE, '?'):<12}{v.tags.get(TAG_TRAINED_VIA, '?'):<12}{v.run_id}"
        )
