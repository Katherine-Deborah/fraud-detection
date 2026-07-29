"""Shared helper for constructing a Feast FeatureStore whose online-store
connection string can be overridden by environment, without touching
feature_store.yaml (which stays correct for host-venv scripts).

feature_store.yaml pins `connection_string: "localhost:6380"` -- correct
when materialize.py/consume.py/check_consistency.py run as plain Windows
processes (see docs/feature_store.md's WSL port-collision writeup). Inside
the Airflow container (Session 4), "localhost" refers to that container
itself, not the redis service; the compose network's own DNS name (`redis`,
port 6379, no collision there since it never crosses the Windows host
loopback) is reachable instead. Rather than fork feature_store.yaml per
environment, this loads the repo's config and overrides just that one field
when FEAST_REDIS_CONNECTION_STRING is set.
"""

from __future__ import annotations

import os
from pathlib import Path

from feast import FeatureStore
from feast.repo_config import load_repo_config

FEAST_REDIS_CONNECTION_STRING_ENV = "FEAST_REDIS_CONNECTION_STRING"


def get_feature_store(repo_path: Path | str) -> FeatureStore:
    repo_path = Path(repo_path)
    config = load_repo_config(repo_path, repo_path / "feature_store.yaml")
    override = os.environ.get(FEAST_REDIS_CONNECTION_STRING_ENV)
    if override:
        config.online_store.connection_string = override
    # repo_path must still be passed alongside config: FeatureStore resolves
    # the registry's relative path (registry.db) against repo_path, and
    # silently falls back to os.getcwd() if only config is given -- which
    # pointed at the wrong (empty) registry and made every push fail with
    # PushSourceNotFoundException even though `feast apply` had been run.
    return FeatureStore(repo_path=repo_path, config=config)
