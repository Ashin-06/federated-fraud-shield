"""The model service: settings, auth, limits, headers, metrics, health, the model registry,
the score log and migrations."""
import hashlib
import importlib
import json
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from fedshield import db, ops
from fedshield.data import FEATURES, consortium

KEY = "fedshield_" + "x" * 40
HASHED = {f"sha256:{hashlib.sha256(KEY.encode()).hexdigest()}": {"user": "risk", "role": "scorer"}}
ROWS = consortium(0)[0].x_test[:5].tolist()


def boot(monkeypatch, **env):
    for k in [k for k in os.environ if k.startswith("FEDSHIELD_")]:
        monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(f"FEDSHIELD_{k}", v)
    import fedshield.api as api

    return importlib.reload(api)


@pytest.mark.parametrize("change, message", [
    ({"FEDSHIELD_API_KEYS": None}, "API keys"),
    ({"FEDSHIELD_DATABASE_URL": None}, "DATABASE_URL"),
    ({"FEDSHIELD_API_KEYS": json.dumps({KEY: {"user": "r", "role": "scorer"}})}, "sha256"),
    ({"FEDSHIELD_CORS_ORIGINS": "*"}, "CORS"),
])
def test_production_settings_refused(change, message):
    env = {"FEDSHIELD_ENV": "production", "FEDSHIELD_API_KEYS": json.dumps(HASHED), "FEDSHIELD_DATABASE_URL": "sqlite://"}
    assert ops.load_settings(env).production
    env.update(change)
    with pytest.raises(ops.ConfigError, match=message):
        ops.load_settings({k: v for k, v in env.items() if v is not None})


def test_no_public_bind_and_readable_exit():
    with pytest.raises(ops.ConfigError, match="without API keys"):
        ops.load_settings({"FEDSHIELD_HOST": "0.0.0.0"})
    env = {k: v for k, v in os.environ.items() if not k.startswith("FEDSHIELD_")}
    r = subprocess.run([sys.executable, "-m", "fedshield", "serve", "--host", "0.0.0.0"], env=env,
                       capture_output=True, text=True, timeout=300, check=False)
    assert r.returncode != 0 and "without API keys" in r.stderr


def test_models_are_published_scored_and_kept_across_a_restart(monkeypatch, tmp_path):
    url = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'fed.db'}"
    c = TestClient(boot(monkeypatch, DATABASE_URL=url).app)
    assert c.post("/v1/score", json={"rows": ROWS}).status_code == 409  # nothing published yet
    assert c.get("/readyz").status_code == 200  # ready before a model exists, so /v1/train can reach it
    v = c.post("/v1/train", json={"rounds": 2, "dp_sigma": 1.0}).json()
    assert v["version"] >= 1 and v["epsilon"] is not None
    assert c.post("/v1/train", json={"rounds": 1, "secure_agg": True, "aggregator": "median"}).status_code == 422
    first = c.post("/v1/score", json={"rows": ROWS}).json()

    c = TestClient(boot(monkeypatch, DATABASE_URL=url).app)  # restart: the model comes back from the database
    again = c.post("/v1/score", json={"rows": ROWS}).json()
    assert again["version"] == first["version"] and again["scores"] == first["scores"]
    assert c.get("/readyz").status_code == 200
    models = c.get("/v1/models").json()
    assert models["features"] == FEATURES and models["models"][0]["config"]["dp_sigma"] == 1.0
    assert c.post("/v1/score", json={"rows": [[1.0, 2.0]]}).status_code == 422
    import fedshield.api as current

    with current.engine.connect() as cx:  # both scoring calls were logged, across the restart
        assert cx.execute(text("SELECT COUNT(*) FROM score_log")).scalar() >= 2


def test_roles_and_metrics(monkeypatch):
    keys = {"k-score": {"user": "risk", "role": "scorer"}, "k-train": {"user": "ml", "role": "trainer"},
            "k-mon": {"user": "prom", "role": "monitor"}}
    c = TestClient(boot(monkeypatch, API_KEYS=json.dumps(keys)).app)
    assert c.post("/v1/train", json={"rounds": 1}).status_code == 401
    assert c.post("/v1/train", json={"rounds": 1}, headers={"x-api-key": "k-score"}).status_code == 403
    assert c.post("/v1/train", json={"rounds": 1}, headers={"x-api-key": "k-train"}).status_code == 200
    assert c.post("/v1/score", json={"rows": ROWS}, headers={"x-api-key": "k-train"}).status_code == 403
    assert c.post("/v1/score", json={"rows": ROWS}, headers={"x-api-key": "k-score"}).status_code == 200
    m = c.get("/metrics", headers={"x-api-key": "k-mon"})
    assert m.status_code == 200 and 'fedshield_decisions_total{decision="model_published"}' in m.text


def test_limits_headers_errors_and_health(monkeypatch):
    c = TestClient(boot(monkeypatch, RATE_LIMIT="3").app)
    codes = [c.get("/v1/models").status_code for _ in range(5)]
    assert codes[:3] == [200] * 3 and codes[3:] == [429, 429]
    api = boot(monkeypatch, MAX_BODY_BYTES="300")

    @api.app.get("/boom")
    def boom():
        raise RuntimeError("secret detail: bank ledger 4111")

    c = TestClient(api.app)
    assert c.post("/v1/score", json={"rows": ROWS}).status_code == 413
    crash = c.get("/boom")
    assert crash.status_code == 500 and "4111" not in crash.text and crash.json()["request_id"]
    bad = c.post("/v1/train", json={"rounds": "SECRET"})
    assert bad.status_code == 422 and "SECRET" not in bad.text
    r = c.get("/v1/models", headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123" and r.headers["content-security-policy"] == ops.API_CSP
    assert c.get("/healthz").json() == {"status": "ok"}


def test_migrations_apply_once(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    eng = db.engine(url)
    assert db.migrate(eng) == ["001_init.sql"] and db.migrate(eng) == []
    with eng.connect() as c:
        tables = {r[0] for r in c.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))}
    assert {"models", "score_log", "schema_migrations"} <= tables
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("FEDSHIELD_")}, "FEDSHIELD_DATABASE_URL": url}
    out = subprocess.run([sys.executable, "-m", "fedshield", "migrate"], env=env, capture_output=True, text=True,
                         timeout=300, check=False)
    assert out.returncode == 0 and "up to date" in out.stdout
