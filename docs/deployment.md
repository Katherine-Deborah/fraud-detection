# Deployment (Session 10)

Three deliverables, matching SESSIONS.md's Session 10 task list: a
finalized `docker-compose.yml` that brings up every service with
`depends_on`+healthchecks (not just declaration order), Kubernetes
manifests for the serving layer tested against a real local cluster, and a
narrow Terraform module for the SageMaker+S3 pieces from Session 6.

## Tooling installed this session

Neither `kind` nor `terraform` were on this machine, and `choco install`
needed admin rights this session didn't have (it also tried to pull in a
`docker-desktop` package as a dependency of `kind`, which risked disrupting
the already-running stack). Both were downloaded directly as standalone
binaries into `%USERPROFILE%\.local\bin` instead — no elevation, no touching
the existing Docker Desktop install:

- `kind` v0.31.0 (from `kind.sigs.k8s.io`)
- `terraform` v1.9.8 (from `releases.hashicorp.com`)

Add that directory to `PATH`, or call the binaries by full path (as this
session's commands and `k8s/README.md`/`infra/terraform/README.md` do).

## Docker Compose finalization

**New service: `fastapi-gateway`.** The FastAPI gateway (`serving/app.py`)
ran as a local host process through Sessions 8/9 (documented reason at the
time: an isolated venv, `feast` needing `numpy>=2` vs. `mlflow` needing
`numpy<2`, made a container not worth fighting for). Now containerized via
`Dockerfile.serving`'s `dev` build target, following the same split
`Dockerfile.airflow` already uses: the image bakes in only
`requirements-serving.txt`; `serving/`, `feature_store/`,
`data_generation/` are bind-mounted so a model re-export or `feast apply`
is picked up on container restart with no rebuild.

**Two bugs found only by actually starting the container, not by reading
the Dockerfile:**

1. `serving/run_server.py` wipes and recreates a Prometheus multiprocess
   directory (`serving/_prom_multiproc`) on every startup — inside the
   bind-mounted `./serving`, that directory can carry file permissions
   from a *previous* run (a different container, or a host process) that
   the new container's user can't `rmtree` through Docker Desktop's file-
   sharing layer, crashing on `PermissionError: counter_NNNNN.db` before
   the app ever started. Fixed by making the multiproc directory
   overridable via `PROMETHEUS_MULTIPROC_DIR` (falls back to the old
   `serving/_prom_multiproc` default for host-process runs, unchanged);
   `docker-compose.yml` points the container at `/tmp/prom_multiproc`
   instead — container-local, never bind-mounted, so this can't recur.
2. **A stale host-process gateway from a previous session was still
   running and still bound to port 8090** (`serving/run_server.py`
   under `.venv-serving`, PID tree confirmed via `Get-CimInstance
   Win32_Process`), racing the new container for the same published port.
   Docker Desktop's Windows networking let both bindings coexist well
   enough that curl requests were nondeterministically answered by
   whichever one happened to be listening, which showed up as `/metrics`
   reporting request counts left over from Session 9's load test on a
   container that had just started. Killed the stale process tree (safe
   per Session 9's own session-log note: "safe to restart, no state
   depends on them staying up between sessions") before re-verifying.

**Wiring:** `TRITON_HTTP_URL` in `serving/app.py` was hardcoded to
`http://localhost:8000` — inside the gateway's own container that means
the gateway itself, not the `triton` service. Made overridable via env
(`TRITON_HTTP_URL`), same pattern already used for
`FEAST_REDIS_CONNECTION_STRING`. `docker-compose.yml` sets both to the
compose service DNS names for the `fastapi-gateway` service.
`monitoring/prometheus/prometheus.yml`'s `fraud-gateway` scrape target
changed from `host.docker.internal:8090` (reaching out to the host process)
to `fastapi-gateway:8090` (a real compose service now) — confirmed via
Prometheus's own `/api/v1/targets` API that both `fraud-gateway` and
`triton` scrape targets report `health: up` after the change (Prometheus
does **not** pick up a mounted config file change on its own; it needed an
explicit `docker compose restart prometheus`, since compose only recreates
a container when the *service definition* changes, not when a bind-mounted
file's contents change).

**Dependency graph:** `fastapi-gateway` now `depends_on` `redis` and
`triton` (both `condition: service_healthy`); `prometheus` now
`depends_on` `triton` and `fastapi-gateway` (both healthy) instead of
having no explicit dependency on the thing it scrapes. Everything else
(Kafka, Postgres, Airflow, MLflow) already had healthchecks/`depends_on`
from Sessions 2–9; producer/consumer deliberately stay local scripts, not
compose services (they're one-shot demo triggers, not long-running infra —
matches how this file has drawn that line since Session 2).

**Verified end-to-end, live:** `docker compose build fastapi-gateway` →
`docker compose up -d fastapi-gateway prometheus` → both reached healthy
→ `/predict` against a real account returned the identical fraud score
(`0.00294...` for `acct_000000`) the host-process gateway produced in
Session 8 → `/metrics` correctly counted from zero on a fresh container →
Prometheus scraping both targets successfully.

**Added `.dockerignore`** (repo root, didn't exist before) — without it,
every `docker compose build` was sending the multi-gigabyte
`.venv*`/`data/` directories to the daemon on every build even though
neither `Dockerfile.airflow` nor `Dockerfile.serving` actually `COPY`s
them.

## Kubernetes (serving layer only)

Scope matches PROJECT.md §5.9 literally — **only the FastAPI gateway** is
deployed into the cluster (`k8s/serving-deployment.yaml`,
`k8s/serving-service.yaml`). Redis and Triton are not redeployed; the
Deployment reaches the same containers `docker-compose.yml` already runs,
via `host.docker.internal`. Full narrow-scope reasoning, the exact
commands to reproduce this, and "how this would differ in a real cluster"
are in `k8s/README.md` — this section covers only what went wrong and how
it was diagnosed, since that's the part worth remembering.

**Two real bugs, both found only by watching pods actually fail, not by
reading the YAML:**

1. First deploy: every pod `CrashLoopBackOff`, readiness/liveness probes
   both `connection refused`. Root cause: `host.docker.internal` resolves
   fine *inside the kind node container itself* (confirmed via
   `docker exec fraud-detection-control-plane getent ahostsv4
   host.docker.internal` → `192.168.65.254`), but a **pod** runs in its
   own CNI network namespace (kindnet) with no `/etc/hosts`/DNS entry for
   that name at all — the app was hanging trying to connect Feast's Redis
   client to an unresolvable host, so uvicorn never got far enough to
   bind its own port. Fixed with an explicit `hostAliases` entry in the
   pod spec, pinned to the IP the node itself resolves that name to.
2. Second deploy (after the fix above): pods now started, but got
   `OOMKilled` under the original `512Mi` limit. Cause: `run_server.py`
   defaults to `--workers 4` (Session 8/9's tuning for a *single* host
   process), and 2 replicas × 4 workers/pod meant 8 full uvicorn
   processes each loading pandas/numpy/feast independently. Fixed two
   ways: made worker count configurable
   (`UVICORN_WORKERS` env, `serving/run_server.py`), set to `1` in the
   Deployment (k8s scales via replica count, not stacking workers inside
   one pod — that's the actual point of a Deployment), and raised the
   memory limit to `768Mi` as a margin on top of that fix.

**A second Docker build target was needed**, not the same image compose
uses: k8s has no portable equivalent of a host bind mount (`hostPath` is
node-local, not something a Deployment should depend on), so
`Dockerfile.serving`'s new `k8s` target `COPY`s `serving/`,
`feature_store/`, `data_generation/` — including the generated artifacts a
fresh checkout wouldn't have (`feature_store/feature_repo/registry.db`,
`serving/model_metadata.json`) — into a self-contained image, loaded into
the kind cluster via `kind load docker-image` (no registry involved).

**Verified end-to-end, live:** both pods reached `1/1 Running` with **zero
restarts** after both fixes; `kubectl port-forward svc/fraud-fastapi-gateway`
+ the same `/predict` call returned the identical fraud score the
docker-compose gateway produced — confirming the k8s code path agrees with
compose, not just that the pod started.

## Terraform (SageMaker + S3 only)

`infra/terraform/` — `s3.tf` (the Session 6 training-data bucket, now with
its lifecycle rule widened to the *whole* bucket, closing a gap
`docs/sagemaker.md` explicitly flagged as unfixed: Session 6's script-created
rule only covered `training-data/`, leaving `code/`/`output/` to
accumulate indefinitely) and `iam.tf` (the SageMaker execution role, reusing
the exact trust/permissions JSON already tested live in Session 6 via
`file()`, not a re-authored copy). Full scope reasoning (why the CLI user
and the training job itself are deliberately **not** in this module) is in
`infra/terraform/README.md`.

Per this session's confirmed plan: **written, `init`'d, `validate`'d, and
`plan`'d live against real AWS credentials — never `apply`'d.**
`terraform plan -var="region=us-east-1"` came back clean: 5 resources to
add, 0 changed, 0 destroyed, no errors, real IAM/S3 API calls succeeding
against the scoped CLI user's credentials.

**One important finding, not theoretical:** the plan's computed
`bucket_name` output was `fraud-detection-sagemaker-183079729790` — the
**exact bucket Session 6 already created by hand**. Applying this module
today, as written, would collide with real, already-existing infrastructure
rather than create anything fresh. Documented prominently in
`infra/terraform/README.md` with the two ways to resolve it (import the
existing resources, or point this module at a different name) — flagging
this now, before it becomes a surprise mid-`apply`, is the entire reason
this session stopped at `plan` rather than going further.

**One pre-existing `.gitignore` bug fixed along the way:** `infra/**/.terraform.lock.hcl`
was excluded from version control — the opposite of Terraform's own
documented recommendation (the lock file is what makes `terraform init`
reproduce the exact same provider version later). It predates any real
Terraform config existing in this repo (added defensively in an earlier
session); fixed now that there's a real lock file to test the point
against.

**`infra/terraform/destroy_reminder.py`** — the required "destroy
reminder/script." Wraps `terraform destroy` behind a typed `DESTROY`
confirmation (not Terraform's own reflexive y/n), then re-runs the
`list-endpoints`/`list-notebook-instances` safety check every prior
AWS-touching session ran by hand in its session log — but across **every**
region this project has ever touched (`us-east-1`, `us-west-2`,
`ap-southeast-2`), not just the one being destroyed, since a resource
stranded in a region nobody's currently looking at is the actual risk.
Ran its safety-check half live this session (`--skip-destroy`, since
nothing was ever applied): clean in all three regions.

## Cost/safety check

No AWS resources were created, modified, or destroyed this session —
`terraform plan` and `destroy_reminder.py --skip-destroy` are both
read-only. `destroy_reminder.py`'s live run this session confirmed:
`list-endpoints` and `list-notebook-instances` both empty in `us-east-1`,
`us-west-2`, and `ap-southeast-2`. AWS spend this session: $0.

## What's left running at session end

Kafka, Redis, Postgres, Airflow, MLflow, Triton, Prometheus, Grafana, and
now `fastapi-gateway` — all via `docker compose up -d`, all safe to
restart. The kind cluster (`fraud-detection`) was deleted at the end of
this session (`kind delete cluster --name fraud-detection`) since it's
fully reproducible from `k8s/README.md`'s documented steps and there's no
reason to leave a second Kubernetes control plane idling on this machine
between sessions.

## What I'd try with more time

- The `host.docker.internal` + `hostAliases` approach for reaching
  Redis/Triton from the kind cluster is a local-dev-only shortcut (see
  `k8s/README.md`'s "how this would differ in a real cluster"). A real
  next step would be Deployments/Services for Redis and Triton *inside*
  the cluster, so the serving layer's k8s manifests don't implicitly
  depend on `docker-compose.yml` also being up on the same machine.
- `terraform import` the real Session 6 bucket/role into this module's
  state, so `plan` reflects actual drift instead of "create from
  scratch" — deferred rather than done blind, since importing without
  the user present to confirm which resources is exactly the kind of
  AWS-state-mutating action this project's conventions gate on
  confirmation first.
- A CI workflow that runs `terraform validate`/`kubeval` (or similar) on
  every push, so manifest/module drift is caught before a session
  ever needs to hand-verify it live again.
