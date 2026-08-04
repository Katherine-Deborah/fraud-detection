# Model Registry + Staged Rollout — Session 7

This document describes the 4-stage promotion pipeline
(**Staging → Shadow → Canary → Production**) built on top of the MLflow
Model Registry, the criteria gating each transition, the rollback path,
and a real end-to-end run of the whole thing.

## Why a custom stage tag, not MLflow's built-in "stage"

MLflow's classic model-version `stage` field only has 4 values
(`None`/`Staging`/`Production`/`Archived`) and is itself deprecated
upstream in favor of aliases/tags. Neither has room for this project's
`Shadow` and `Canary` stages. So `training/registry.py` never calls
`transition_model_version_stage`; instead the current stage lives in a
plain model-version tag, `lifecycle_stage`, with 6 possible values:

```
staging -> shadow -> canary -> production
                                   |
                                archived   (superseded by a later promotion,
                                            or manually rolled back into)
staging -> rejected   (failed a gate before ever reaching production)
```

All 5 models trained across Sessions 5-6 (`logistic_regression`,
`random_forest`, `isolation_forest`, `lstm`, `random_forest_sagemaker`) are
registered as **versions of one registered model**,
`fraud-detection-classifier` — they're competing candidates for the same
production slot, not separate products. Every stage transition is also
logged as its own run in a separate `model-registry-audit` MLflow
experiment (params/tags/metrics describing the decision): the
`lifecycle_stage` tag tells you where a version is *now*, the audit
experiment tells you *how it got there and why*. Browse both at
`http://localhost:5000`.

## Traffic source (until Session 8 exists)

There's no live serving layer yet — that's Session 8. Until then, the
shadow and canary harnesses replay the chronological **test split** from
`training/data_prep.py` (the period every model's Session 5/6 metrics were
already computed on, never used to fit or threshold-select any model) as
a stand-in for live traffic. This is documented explicitly rather than
quietly treated as equivalent to real traffic — Session 8's FastAPI
gateway is the natural place to point these harnesses at real requests
later, and the harness functions (`run_shadow`, `run_canary`) are written
to take a client + model version, not a hardcoded data source, so that
swap should be small.

Both models in a comparison score with **each model's own pre-selected
threshold** — the one chosen on validation during that model's own
training run (`metrics.threshold`, logged in Session 5/6), never a
threshold re-picked on the shadow/canary batch itself, which would just be
tuning on test data under a different name.

## Promotion criteria

Full numeric definitions live in `training/promotion_criteria.py`; the
reasoning:

| Gate | What it checks | Threshold | Why |
|---|---|---|---|
| **Staging → Shadow** | Candidate's already-logged test-set AUC-PR vs. current production's | candidate ≥ 90% of production's AUC-PR (or auto-pass if no production exists — bootstrap) | Free — reuses Session 5/6 metrics, no new scoring. Blocks an obviously-worse candidate before paying for a shadow-scoring pass. Not "must beat outright," so a near-parity candidate (e.g. cheaper/simpler) can still get a real look. |
| **Shadow → Canary** | Candidate vs. production recall (both at their own thresholds) on the *same* shadow batch, plus candidate's own FPR budget | candidate recall ≥ production recall − 2pp, and candidate FPR ≤ 2% | Recomputes the comparison on data neither model was tuned on this time — a real head-to-head, not reused offline numbers. The FPR budget is PROJECT.md §6's target; a model that "wins" on recall by flagging everything isn't a real win. |
| **Canary → Production** | Same recall/FPR comparison, but on the live-acted 10% canary slice, plus a minimum sample floor | canary recall ≥ production recall − 2pp, canary FPR ≤ 2.5%, and ≥ 20 fraud cases in the slice | The canary slice is smaller and noisier than the full shadow batch, so the FPR budget is slightly looser and a sample-size floor stops a lucky/unlucky small slice from single-handedly promoting or sinking a candidate. |

If a candidate fails any gate it is tagged `rejected` and stops there —
cheaper gates run first on purpose, so an obviously bad candidate (see
Isolation Forest below) never reaches the more expensive canary stage.

## Rollback path

`training/registry.py::rollback(client, target_version, reason)` is the
generic mechanism: `target_version` (which must currently be `archived`)
becomes the new `production`; whatever is currently `production` (if
anything) is demoted to `archived`. It is **not** tied to a canary
failure — it's callable any time an operator decides the current
production model needs to be reverted, e.g. a problem noticed only after
real traffic exposure. Both sides of the swap are logged as their own
audit events. Because the function is symmetric (it just needs its target
to be `archived`), calling it twice — once to roll back, once to roll
forward again — is how §"Live walkthrough" below demonstrates it working
in both directions.

## Live walkthrough (real run, 2026-08-04)

`training/run_registry_walkthrough.py`, run against the live MLflow
server, produced the following (verbatim from the actual run, not
hand-typed):

1. **Registered all 5 candidates** as versions 1–5 of
   `fraud-detection-classifier`, all starting in `staging`.
2. **Bootstrap**: `logistic_regression` (v1) promoted straight to
   `production` — no predecessor exists yet, so the Staging→Shadow gate is
   skipped by definition for the very first model.
3. **`random_forest` (v2) — Staging → Shadow gate**: AUC-PR 0.6554 vs.
   production's 0.3120 (ratio 2.10, well above the 0.90 minimum) → passed,
   promoted to `shadow`.
4. **`random_forest` (v2) — Shadow harness**: scored all 750,903 test-split
   rows alongside production, predictions logged (not acted on) to
   `data/registry_shadow_logs/random_forest_v2.parquet`. Candidate recall
   0.8628 @ FPR 0.0175 vs. production recall 0.7858 → passed, promoted to
   `canary`.
5. **`random_forest` (v2) — Canary harness**: 77,595 rows (10.3% of
   accounts, hash-split) acted on by the candidate — 169 fraud cases,
   above the 20-case floor. Candidate recall 0.8817 @ FPR 0.0178 vs.
   production recall 0.7863 on the other 673,308-row slice → passed.
   `random_forest` (v2) promoted to `production`; `logistic_regression`
   (v1) archived.
6. **`isolation_forest` (v3) — Staging → Shadow gate**: AUC-PR 0.0497 vs.
   the new production's 0.6554 (ratio 0.08, far below 0.90) → **failed**,
   rejected immediately. No shadow or canary compute was spent on it —
   exactly the point of running the cheapest gate first.
7. **`lstm` (v4) and `random_forest_sagemaker` (v5)** registered as
   additional candidates and left in `staging`, not walked through this
   session's live demo:
   - The LSTM's pyfunc interface expects per-account sequence windows
     (`seq_len=20` sliding windows, see `training/train_lstm.py`), not the
     flat feature rows the sklearn-based shadow/canary harnesses score —
     scoring it would need a separate sequence-aware harness.
   - `random_forest_sagemaker` was trained inside the SageMaker container's
     scikit-learn 1.2.1, which Session 6 already established cannot be
     unpickled under a newer scikit-learn (the tree-node struct changed in
     1.3) — `.venv-registry` is pinned to 1.4.2 to match the *local*
     models, so it deliberately never attempts to load this one (see
     `requirements-registry.txt`).
   - Both remain valid future candidates; scoring either would be natural
     follow-up work, not deferred for a quality reason.
8. **Rollback demo**: `registry.rollback(client, logistic_regression, ...)`
   simulating a post-promotion incident — `logistic_regression` (v1) back
   to `production`, `random_forest` (v2) demoted to `archived`.
9. **Rollback reversed**: `registry.rollback(client, random_forest, ...)`
   called a second time — `random_forest` (v2) back to `production`,
   `logistic_regression` (v1) back to `archived`. This is the session's
   intended final state: Random Forest remains the recommended production
   candidate per `docs/model_comparison.md`, and the rollback mechanism
   was proven to work in both directions rather than left as untested code.

Final registry state:

| Version | Algorithm | Stage | Trained via |
|---|---|---|---|
| 1 | logistic_regression | archived | local |
| 2 | random_forest | **production** | local |
| 3 | isolation_forest | rejected | local |
| 4 | lstm | staging | local |
| 5 | random_forest_sagemaker | staging | sagemaker |

Every promotion, rejection, and rollback above also exists as its own run
in the `model-registry-audit` MLflow experiment, with the same numbers as
params/metrics — browse it at `http://localhost:5000` for the full audit
trail (10 audit runs from this single script execution: 1 bootstrap
promotion, 3 RF promotions (staging→shadow, shadow→canary, canary→
production), 1 archive, 1 rejection, and 2 rollback calls × 2 events each
= 4 rollback runs).

## Running it yourself

```bash
python -m venv .venv-registry
.venv-registry/Scripts/pip install -r requirements-registry.txt

# full demo, idempotent-ish (re-running creates new versions 6-10 rather
# than reusing 1-5, since create_model_version always creates a new one --
# fine for a demo script, not intended as a production re-run tool)
.venv-registry/Scripts/python training/run_registry_walkthrough.py

# individual stages, once a candidate is registered:
.venv-registry/Scripts/python training/shadow_harness.py --algorithm random_forest
.venv-registry/Scripts/python training/canary_harness.py --algorithm random_forest
```

## Environment note

`training/registry.py`, `shadow_harness.py`, `canary_harness.py`, and
`run_registry_walkthrough.py` run under a new isolated `.venv-registry`
(`requirements-registry.txt`), not the main `.venv` — same root cause as
Session 6's `.venv-sagemaker`: `mlflow==2.12.2` hard-pins `numpy<2`/
`pyarrow<16`, which conflicts with `feast==0.65.0`'s `numpy>=2.0`
requirement already installed in the main venv. These scripts never import
feast/kafka code, so the older numpy/pyarrow here are harmless.
`.venv-registry` pins `scikit-learn==1.4.2` (matching the *local* Session 5
models) rather than `.venv-sagemaker`'s `1.2.1` (matching the SageMaker
container) — see the "not walked through" note on `random_forest_sagemaker`
above for why that matters.

## What I'd try with more time

- A sequence-aware shadow/canary path for the LSTM (windowed pyfunc
  scoring), so it can actually compete for production rather than sitting
  in `staging` indefinitely.
- Wire the shadow/canary harnesses to Session 8's real serving traffic
  instead of a test-split replay, once that layer exists.
- A scikit-learn-version-agnostic way to compare the SageMaker-trained
  model against the others without needing a matching interpreter (e.g.
  export both to ONNX and score via a common runtime — also sets up
  Session 8's Triton export step).
- Statistical significance testing on the canary comparison (currently a
  fixed-tolerance point comparison) — meaningful once traffic volume is
  real rather than a fixed historical replay.
