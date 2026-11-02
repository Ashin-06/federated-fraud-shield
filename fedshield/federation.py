"""Federated training loop, baselines and evaluation."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from . import secure_agg
from .aggregators import AGGREGATORS
from .attacks import poison_data, poison_update
from .data import TYPOLOGIES, Bank, apply_trigger
from .model import FraudNet, flat, load_flat, predict, train_local
from .privacy import Accountant


@dataclass
class FedConfig:
    rounds: int = 30
    local_epochs: int = 1
    lr: float = 0.02
    dp_lr: float = 0.5  # DP-SGD needs a larger step: clipped, noisy gradients are small
    batch: int = 64
    aggregator: str = "fedavg"
    attack: str = "none"
    n_malicious: int = 0
    boost: float | None = None  # default: number of clients in the round (model replacement)
    dp_sigma: float = 0.0
    dp_clip: float = 1.0
    delta: float = 1e-5
    secure_agg: bool = False
    client_clip: float | None = None
    clients_per_round: int | None = None
    seed: int = 0


@dataclass
class RoundStats:
    round: int
    pr_auc: float
    per_bank: dict
    recall_by_typology: dict
    backdoor_asr: float
    epsilon: float


@dataclass
class History:
    config: dict
    rounds: list[RoundStats] = field(default_factory=list)
    malicious: list[int] = field(default_factory=list)

    @property
    def final(self) -> RoundStats:
        return self.rounds[-1]


def federated_stats(banks: list[Bank]) -> tuple[np.ndarray, np.ndarray]:
    """Global feature mean/std from per-bank sums: no bank reveals its rows
    (and the sums themselves can go through secure aggregation)."""
    n = sum(len(b.x_train) for b in banks)
    s = sum(b.x_train.sum(0, dtype=np.float64) for b in banks)
    ss = sum((b.x_train.astype(np.float64) ** 2).sum(0) for b in banks)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean**2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def _pos_weight(y: np.ndarray) -> float:
    p = y.mean()
    return float(np.clip((1 - p) / max(p, 1e-6), 1.0, 30.0))


def evaluate(model, tests, norm) -> dict:
    xs = np.vstack([t[0] for t in tests])
    ys = np.concatenate([t[1] for t in tests])
    typ = np.concatenate([t[2] for t in tests])
    p = predict(model, xs)
    thr = np.quantile(p[ys == 0], 0.99)  # operate at 1% false-positive rate
    per_bank, i = {}, 0
    for _x, y, _, name, _raw in tests:
        per_bank[name] = float(average_precision_score(y, p[i : i + len(y)]))
        i += len(y)
    recall = {t: float((p[typ == t] > thr).mean()) for t in sorted(set(typ) - {""})}  # typologies present in the test data
    raw_fraud = np.vstack([t[4][t[1] == 1] for t in tests])
    trig = predict(model, norm(apply_trigger(raw_fraud)))
    return {"pr_auc": float(average_precision_score(ys, p)), "per_bank": per_bank, "recall": recall,
            "backdoor_asr": float((trig <= thr).mean())}


def _prepare(banks: list[Bank]):
    mean, std = federated_stats(banks)
    def norm(x):
        return ((x - mean) / std).astype(np.float32)

    tests = [(norm(b.x_test), b.y_test, b.typ_test, b.name, b.x_test) for b in banks]
    return norm, tests


def run(banks: list[Bank], cfg: FedConfig = FedConfig(), on_round=None) -> History:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    norm, tests = _prepare(banks)
    n_clients = len(banks)
    malicious = set(rng.choice(n_clients, cfg.n_malicious, replace=False).tolist()) if cfg.attack != "none" else set()
    data = []
    clean_sizes = []
    for i, b in enumerate(banks):
        x, y = norm(b.x_train), b.y_train
        clean_sizes.append(len(x))
        if i in malicious:
            x, y = poison_data(cfg.attack, x, y, rng)
        data.append((x, y))

    if cfg.secure_agg and cfg.aggregator != "fedavg":
        raise ValueError("secure aggregation only reveals the sum; use aggregator='fedavg' (optionally with client_clip)")
    sec_clients = [secure_agg.SecAggClient(i) for i in range(n_clients)] if cfg.secure_agg else None

    model = FraudNet(len(banks[0].x_train[0]))
    accountants = [Accountant(cfg.dp_sigma, min(1.0, cfg.batch / len(d[0])), cfg.delta) for d in data]
    hist = History(asdict(cfg), malicious=sorted(malicious))
    per_round = cfg.clients_per_round or n_clients
    boost = cfg.boost or float(per_round)

    for r in range(cfg.rounds):
        chosen = sorted(rng.choice(n_clients, per_round, replace=False).tolist())
        w0 = flat(model)
        updates, ns = [], []
        for c in chosen:
            local = copy.deepcopy(model)
            x, y = data[c]
            steps = train_local(local, x, y, cfg.local_epochs, cfg.dp_lr if cfg.dp_sigma > 0 else cfg.lr, cfg.batch, _pos_weight(banks[c].y_train),
                                cfg.dp_sigma, cfg.dp_clip, seed=cfg.seed * 10_000 + r * 100 + c)
            accountants[c].step(steps)
            u = flat(local) - w0
            if c in malicious:
                u = poison_update(cfg.attack, u, boost, g)
            if cfg.client_clip:
                u = u * min(1.0, cfg.client_clip / (u.norm().item() + 1e-12))
            updates.append(u)
            ns.append(clean_sizes[c])

        n = torch.tensor(ns, dtype=torch.float32)
        if cfg.secure_agg:
            directory = {c: sec_clients[c].public_key for c in chosen}
            total_n = float(n.sum().item())
            masked = [sec_clients[c].masked_update((u * (k / total_n)).numpy(), directory, r) for c, u, k in zip(chosen, updates, ns)]
            agg = torch.as_tensor(secure_agg.aggregate(masked), dtype=torch.float32)
        else:
            agg = AGGREGATORS[cfg.aggregator](torch.stack(updates), n, f=max(1, cfg.n_malicious))
        load_flat(model, w0 + agg)

        m = evaluate(model, tests, norm)
        honest = [a.epsilon for i, a in enumerate(accountants) if i not in malicious]
        stats = RoundStats(r + 1, m["pr_auc"], m["per_bank"], m["recall"], m["backdoor_asr"], max(honest) if honest else 0.0)
        hist.rounds.append(stats)
        if on_round:
            on_round(stats)
    hist.model = model
    return hist


def local_only(banks: list[Bank], epochs: int = 30, seed: int = 0) -> dict:
    """Each bank trains alone and is tested on its own (all-typology) test set."""
    norm, tests = _prepare(banks)
    out = {}
    for i, b in enumerate(banks):
        torch.manual_seed(seed)
        m = FraudNet(b.x_train.shape[1])
        train_local(m, norm(b.x_train), b.y_train, epochs, 0.01, 64, _pos_weight(b.y_train), seed=seed + i)
        x, y, typ = tests[i][0], tests[i][1], tests[i][2]
        p = predict(m, x)
        thr = np.quantile(p[y == 0], 0.99)
        unseen = np.isin(typ, [t for t in TYPOLOGIES if t not in b.known])
        out[b.name] = {"pr_auc": float(average_precision_score(y, p)), "recall_unseen_typologies": float((p[unseen] > thr).mean())}
    return out


def centralized(banks: list[Bank], epochs: int = 30, seed: int = 0) -> dict:
    """Upper bound: pool everyone's raw data (what regulators will not allow)."""
    norm, tests = _prepare(banks)
    x = np.vstack([norm(b.x_train) for b in banks])
    y = np.concatenate([b.y_train for b in banks])
    torch.manual_seed(seed)
    m = FraudNet(x.shape[1])
    train_local(m, x, y, epochs, 0.01, 64, _pos_weight(y), seed=seed)
    return evaluate(m, tests, norm)


def per_bank_federated(hist: History, banks: list[Bank]) -> dict:
    norm, tests = _prepare(banks)
    out = {}
    for (x, y, typ, name, _), b in zip(tests, banks):
        p = predict(hist.model, x)
        thr = np.quantile(p[y == 0], 0.99)
        unseen = np.isin(typ, [t for t in TYPOLOGIES if t not in b.known])
        out[name] = {"pr_auc": float(average_precision_score(y, p)), "recall_unseen_typologies": float((p[unseen] > thr).mean())}
    return out
