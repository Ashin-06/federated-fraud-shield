"""Model service for the federated fraud model.

    POST /v1/train       run a federation on the simulated consortium and publish a model version [train]
    GET  /v1/models      published versions with their metrics and privacy budget [read]
    POST /v1/score       fraud scores from the newest version, flagged at its 1% false-positive threshold [score]
    GET  /healthz, /readyz, /metrics

Training here runs the whole consortium in this process on generated bank data, as the CLI does;
a real consortium trains each bank in its own service (see the README). Model versions are stored
as float32 arrays in the database, never as pickles. Settings: ops.py and docs/runbook.md.

    python -m fedshield serve
"""
from __future__ import annotations

import base64
import json
import math
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from . import db, ops
from .data import FEATURES, consortium
from .federation import FedConfig, federated_stats, run
from .model import FraudNet, flat, load_flat, predict

settings = ops.load_settings()
ops.setup_logging(settings.log_level)
engine = db.connect(settings.database_url)
_train_lock = threading.Lock()  # training is CPU-heavy: one at a time

app = FastAPI(title="Federated fraud model service", version="1.0.0")


def _arr(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()


def _unarr(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4")


def newest() -> Optional[Dict[str, Any]]:
    with engine.connect() as c:
        r = c.execute(text("SELECT * FROM models ORDER BY version DESC LIMIT 1")).mappings().first()
    return dict(r) if r else None


_cache: Dict[str, Any] = {}


def _model(row: Dict[str, Any]):
    """The newest version as a ready model, rebuilt only when the version changes."""
    if _cache.get("version") != row["version"]:
        m = FraudNet(int(row["d_in"]))
        load_flat(m, torch.as_tensor(_unarr(row["weights"]).copy()))
        _cache.update(version=row["version"], model=m, mean=_unarr(row["mean"]), std=_unarr(row["std"]),
                      threshold=float(row["threshold"]))
    return _cache


# Ready means the database answers. Not "a model is published": that would keep traffic, including the
# /v1/train call that publishes the first model, away from a new deployment. /v1/score returns 409 until then.
ops.install(app, settings, {"database": lambda: engine.connect().close()})


class TrainRequest(BaseModel):
    rounds: int = Field(30, ge=1, le=60)
    aggregator: Literal["fedavg", "median", "trimmed_mean", "krum", "multi_krum", "norm_clip"] = "fedavg"
    dp_sigma: float = Field(0.0, ge=0, le=10, allow_inf_nan=False)
    secure_agg: bool = False
    client_clip: Optional[float] = Field(None, gt=0, le=100, allow_inf_nan=False)
    seed: int = Field(0, ge=0, le=2**31 - 1)


class ScoreRequest(BaseModel):
    rows: List[List[float]] = Field(..., min_length=1, max_length=1000,
                                    description=f"each row: the {len(FEATURES)} features in data.FEATURES order")


@app.post("/v1/train")
def train(req: TrainRequest, ident: dict = Depends(ops.require("train"))) -> Dict[str, Any]:
    if req.secure_agg and req.aggregator != "fedavg":
        raise HTTPException(422, "secure aggregation only works with fedavg")
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(409, "a training run is already in progress")
    try:
        banks = consortium(req.seed)
        cfg = FedConfig(rounds=req.rounds, aggregator=req.aggregator, dp_sigma=req.dp_sigma, secure_agg=req.secure_agg,
                        client_clip=req.client_clip, seed=req.seed)
        hist = run(banks, cfg)
        mean, std = federated_stats(banks)
        legit = np.vstack([b.x_test[b.y_test == 0] for b in banks])
        thr = float(np.quantile(predict(hist.model, ((legit - mean) / std).astype(np.float32)), 0.99))
        final = hist.final
        eps = final.epsilon
        metrics = {"pr_auc": final.pr_auc, "per_bank": final.per_bank, "recall_by_typology": final.recall_by_typology,
                   "backdoor_asr": final.backdoor_asr}
        with engine.begin() as c:
            version = (c.execute(text("SELECT COALESCE(MAX(version), 0) FROM models")).scalar() or 0) + 1
            c.execute(text("INSERT INTO models VALUES (:v, :t, :u, :c, :m, :e, :d, :w, :mu, :sd, :th)"),
                      {"v": version, "t": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                       "u": ident["user"] or "demo", "c": json.dumps(req.model_dump()), "m": json.dumps(metrics),
                       "e": "none" if not eps or math.isinf(eps) else f"{eps:.3f}", "d": len(FEATURES),
                       "w": _arr(flat(hist.model).numpy()), "mu": _arr(mean), "sd": _arr(std), "th": f"{thr:.8f}"})
    finally:
        _train_lock.release()
    ops.DECISIONS.labels(decision="model_published").inc()
    return {"version": version, "pr_auc": round(final.pr_auc, 4), "epsilon": None if not eps or math.isinf(eps) else round(eps, 3)}


@app.get("/v1/models", dependencies=[Depends(ops.require("read"))])
def models(limit: int = Query(20, ge=1, le=200)) -> Dict[str, Any]:
    with engine.connect() as c:
        rows = c.execute(text("SELECT version, created_at, trained_by, config, metrics, epsilon, threshold FROM models "
                              "ORDER BY version DESC LIMIT :n"), {"n": limit}).mappings().all()
    return {"features": FEATURES, "models": [{**dict(r), "config": json.loads(r["config"]), "metrics": json.loads(r["metrics"])}
                                             for r in rows]}


@app.post("/v1/score")
def score(req: ScoreRequest, ident: dict = Depends(ops.require("score"))) -> Dict[str, Any]:
    row = newest()
    if row is None:
        raise HTTPException(409, "no model has been published yet")
    if any(len(r) != len(FEATURES) for r in req.rows):
        raise HTTPException(422, f"each row needs {len(FEATURES)} features")
    x = np.asarray(req.rows, dtype=np.float32)
    if not np.isfinite(x).all():
        raise HTTPException(422, "features must be finite numbers")
    m = _model(row)
    p = predict(m["model"], ((x - m["mean"]) / m["std"]).astype(np.float32))
    flagged = p > m["threshold"]
    with engine.begin() as c:
        c.execute(text("INSERT INTO score_log VALUES (:i, :t, :u, :v, :n, :f)"),
                  {"i": uuid.uuid4().hex, "t": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                   "u": ident["user"] or "demo", "v": row["version"], "n": len(p), "f": int(flagged.sum())})
    ops.DECISIONS.labels(decision="flagged").inc(int(flagged.sum()))
    ops.DECISIONS.labels(decision="cleared").inc(int((~flagged).sum()))
    return {"version": row["version"], "threshold": m["threshold"],
            "scores": [round(float(v), 6) for v in p], "flagged": [bool(f) for f in flagged]}
