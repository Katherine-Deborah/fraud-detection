"""Session 9: launches the FastAPI gateway with the Prometheus multiprocess
directory set up correctly.

`uvicorn --workers 4` (docs/serving.md's Session 8 latency finding) forks 4
separate OS processes. prometheus_client's multiprocess mode requires
PROMETHEUS_MULTIPROC_DIR to point at an *empty* directory before any worker
process imports the client library, and that directory must be wiped exactly
once by the parent -- not by each worker independently, since they'd race
and delete each other's just-written files on every restart. Plain
`python -m uvicorn serving.app:app --workers 4` has no parent-process hook to
do that wipe, so this script does it, sets the env var, and only then hands
off to uvicorn (whose worker subprocesses inherit the env var already set).

Usage (replaces the Session 8 command in docs/serving.md):
    .venv-serving/Scripts/python serving/run_server.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Session 10: containerized (docker-compose.yml's fastapi-gateway service)
# runs bind-mount serving/ from the host, and this directory's previous
# contents (written by a prior host-process run, or by a different
# container's PID range) can end up with permissions the container's user
# can't rmtree through Docker Desktop's file-sharing layer -- seen directly
# as `PermissionError: counter_NNNNN.db` on startup. A container-internal
# path (unset host bind mount covering it) sidesteps that entirely; the
# default stays serving/_prom_multiproc for host-process runs (unchanged
# since Session 9).
MULTIPROC_DIR = Path(os.environ.get("PROMETHEUS_MULTIPROC_DIR", str(Path(__file__).resolve().parent / "_prom_multiproc")))


def main() -> None:
    if MULTIPROC_DIR.exists():
        shutil.rmtree(MULTIPROC_DIR)
    MULTIPROC_DIR.mkdir(parents=True)
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(MULTIPROC_DIR)

    # `--workers` spawns each worker as a genuinely separate python.exe
    # process (multiprocessing's Windows "spawn" start method). Contrary to
    # what you'd expect, that does *not* mean each worker rebuilds sys.path
    # from PYTHONPATH at startup: spawn pickles the *parent's current*
    # sys.path into the worker's prep-data and the worker overwrites its own
    # sys.path with that copy verbatim (see multiprocessing.spawn), so
    # setting os.environ["PYTHONPATH"] here has no effect by itself -- it's
    # this process's own sys.path, mutated in-memory *before* uvicorn forks
    # the workers, that actually propagates. Running `python -m uvicorn ...`
    # (docs/serving.md's original Session 8 command) happened to work
    # because `-m` puts the invoking cwd on sys.path for the process itself;
    # plain `python serving/run_server.py` instead puts serving/ (this
    # script's own directory) on sys.path[0], so `serving.app` isn't
    # importable in this process -- or therefore in its spawned workers --
    # without this explicit insert.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import uvicorn

    # 4 workers by default -- the number Session 8's load testing tuned this
    # for on a single host process. Session 10's k8s Deployment overrides
    # this down to 1: with 2 replicas already providing parallelism (and a
    # real cluster scaling replica count, not per-pod worker count, to add
    # capacity), 4 workers per pod meant 8 total uvicorn processes each
    # loading pandas/numpy/feast, which OOM-killed pods at a reasonable
    # memory limit -- see k8s/serving-deployment.yaml.
    workers = int(os.environ.get("UVICORN_WORKERS", "4"))
    uvicorn.run("serving.app:app", host="0.0.0.0", port=8090, workers=workers)


if __name__ == "__main__":
    main()
