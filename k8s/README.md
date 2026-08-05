# Kubernetes manifests — serving layer (Session 10)

Scope is deliberately narrow, matching PROJECT.md §5.9 and the same
narrow-module principle applied to `infra/terraform/`: **only the FastAPI
gateway** is deployed into the cluster here. Redis and Triton are **not**
redeployed — this Deployment reaches the same `redis`/`triton` containers
already running via `docker-compose.yml`, on the host, through
`host.docker.internal`. That's an intentional local-dev shortcut, not a
production pattern — see "How this would differ in a real cluster" below.

## Why `host.docker.internal` needs `hostAliases`

`host.docker.internal` resolves fine from *inside a kind node* (a kind node
is itself a Docker container, and Docker Desktop injects that hostname for
every container). It does **not** resolve from inside a *pod* — pods run in
their own CNI network namespace (kindnet), which has no `/etc/hosts` or DNS
entry for it at all. First deploy attempt failed with every pod stuck in
`CrashLoopBackOff` / readiness-probe `connection refused`, root-caused to
exactly this (the app was never given a chance to bind its port because
Feast's Redis client connect was hanging against an unresolvable host).

Fixed with `hostAliases` in `serving-deployment.yaml`, pinned to the IP
Docker Desktop already resolves `host.docker.internal` to *inside the node
container itself*:

```bash
docker exec fraud-detection-control-plane getent ahostsv4 host.docker.internal
# 192.168.65.254  STREAM host.docker.internal   <- what's in the manifest today
```

If this manifest ever fails to connect again, re-run that command and
update the IP in `serving-deployment.yaml` — it's a Docker Desktop internal
address, not guaranteed stable across machines or Docker Desktop versions.

## Why 1 worker per pod, not 4

`serving/run_server.py` defaults to `--workers 4` (Session 8/9's tuned value
for a single host process). The first deploy attempt, run with that default
and a 512Mi memory limit, got `OOMKilled` — 2 replicas × 4 workers/pod = 8
full uvicorn processes each loading pandas/numpy/feast. Fixed two ways:
`UVICORN_WORKERS=1` in the Deployment env (k8s scales via replica count, not
per-pod worker count — that's the whole point of a Deployment), and the
memory limit raised to 768Mi as a safety margin on top of that fix.

## The `k8s` Docker build target

`docker-compose.yml`'s `fastapi-gateway` service bind-mounts `serving/`,
`feature_store/`, `data_generation/` from the host so code/model changes
don't need a rebuild (see `Dockerfile.serving`'s comment) — fine for a
single-host dev loop, but there's no portable equivalent of a host bind
mount in a real cluster (`hostPath` is node-local). `Dockerfile.serving`'s
`k8s` build target instead `COPY`s that same source, plus the generated
artifacts a fresh checkout wouldn't have
(`feature_store/feature_repo/registry.db`, `serving/model_metadata.json`),
into a self-contained image.

**Rebuild this target whenever the production model changes**
(`training/export_to_onnx.py`) or `feast apply` regenerates the registry —
it's a snapshot, not a live mount.

## Running it yourself

```bash
# 1. Build the self-contained image (needs feature_store/feature_repo/registry.db
#    and serving/model_metadata.json to already exist -- see docs/serving.md)
docker build -f Dockerfile.serving --target k8s -t fraud-detection-fastapi-gateway:k8s .

# 2. Create a local cluster and load the image into it (no registry involved)
kind create cluster --name fraud-detection
kind load docker-image fraud-detection-fastapi-gateway:k8s --name fraud-detection

# 3. Bring up this gateway's dependencies via the existing compose file
docker compose up -d redis triton

# 4. Apply the manifests
kubectl --context kind-fraud-detection apply -f k8s/serving-deployment.yaml -f k8s/serving-service.yaml
kubectl --context kind-fraud-detection rollout status deployment/fraud-fastapi-gateway

# 5. Reach it (ClusterIP -- port-forward for a local demo, see serving-service.yaml)
kubectl --context kind-fraud-detection port-forward svc/fraud-fastapi-gateway 18090:8090
curl -X POST http://localhost:18090/predict -H "Content-Type: application/json" \
  -d '{"account_id":"acct_000000","amount":123.45,"merchant_category":"electronics"}'

# 6. Tear down
kind delete cluster --name fraud-detection
```

Verified live end-to-end this session: 2/2 pods reached `Running` /
`1/1 Ready` with zero restarts after the two fixes above, and a `/predict`
call through the port-forwarded Service returned the identical fraud score
(0.0029... for `acct_000000`) as the docker-compose gateway — confirming the
containerized-vs-k8s code paths agree, not just that the pod started.

## How this would differ in a real (non-local) cluster

- Redis and Triton would be their own Deployments/StatefulSets +
  ClusterIP Services *inside* the cluster (Redis probably a
  StatefulSet for the persistent online-store data; Triton a Deployment
  with a `model-repository` volume, e.g. an EBS/PVC populated by CI).
  `FEAST_REDIS_CONNECTION_STRING`/`TRITON_HTTP_URL` would point at those
  Service DNS names instead of `host.docker.internal` and its
  `hostAliases` workaround, which is specific to "the dependency happens
  to already be running on the machine hosting this local cluster."
- The Service would be a `LoadBalancer` or sit behind an Ingress, not a
  `ClusterIP` reached by `kubectl port-forward`.
- Image would come from a real registry (ECR/GHCR/etc.), not
  `kind load docker-image` — `imagePullPolicy: Never` only makes sense
  for a manually-loaded local image.
- A `HorizontalPodAutoscaler` would replace the fixed `replicas: 2`.
