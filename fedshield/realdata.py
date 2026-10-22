"""The same federation on real card transactions.

ULB / Worldline credit-card fraud data (OpenML #1597): 284,807 real European card
transactions over two days, 492 of them fraud. V1-V28 are PCA components of the
confidential raw fields; Time and Amount are as recorded.

No public dataset comes already split across banks, so the consortium is formed the way
real banks differ: transactions are clustered by behaviour and each cluster is shared out
unevenly (Dirichlet, alpha 0.5), giving every bank its own customer mix, size and fraud
rate. Each bank trains on the first 60% of the two days and is tested on the rest, so the
test is later in time than anything seen in training.

There are no fraud typologies or backdoor trigger fields in this data, so the "unseen
typology" and backdoor measurements of the synthetic suite do not apply here.

    python -m fedshield realdata --data creditcard.pq [--rounds 30]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .data import Bank
from .federation import FedConfig, centralized, local_only, run

N_BANKS = 8


def load(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path) if str(path).endswith((".pq", ".parquet")) else pd.read_csv(path)
    hour = (df["Time"].to_numpy() % 86_400) / 3_600
    x = np.c_[df[[f"V{i}" for i in range(1, 29)]].to_numpy(), np.log1p(df["Amount"].to_numpy()),
              np.sin(hour / 24 * 2 * np.pi), np.cos(hour / 24 * 2 * np.pi)].astype(np.float32)
    return x, df["Class"].astype(int).to_numpy().astype(np.float32), df["Time"].to_numpy()


def consortium(path: str | Path, seed: int = 0, alpha: float = 0.5) -> list[Bank]:
    x, y, t = load(path)
    rng = np.random.default_rng(seed)
    clusters = KMeans(N_BANKS, n_init=4, random_state=seed).fit_predict(x[:, :28])
    owner = np.empty(len(x), dtype=int)
    for k in range(N_BANKS):
        rows = np.flatnonzero(clusters == k)
        owner[rows] = rng.choice(N_BANKS, len(rows), p=rng.dirichlet(np.full(N_BANKS, alpha)))
    cut = np.quantile(t, 0.6)
    banks = []
    for b in range(N_BANKS):
        tr, te = (owner == b) & (t < cut), (owner == b) & (t >= cut)
        banks.append(Bank(f"Bank {b + 1}", x[tr], y[tr], x[te], y[te], np.where(y[te] == 1, "fraud", ""), ("fraud",)))
    return banks


def experiments(path: str | Path, rounds: int = 30, seed: int = 0, log=print) -> dict:
    banks = consortium(path, seed)
    base = dict(rounds=rounds, seed=seed, batch=256, lr=0.05)
    res = {"dataset": "ULB / Worldline card fraud (OpenML #1597), 8 banks by behaviour cluster, time split",
           "banks": [{"bank": b.name, "train_rows": len(b.y_train), "train_fraud": int(b.y_train.sum()),
                      "test_rows": len(b.y_test), "test_fraud": int(b.y_test.sum())} for b in banks]}

    log("1/4 value of federation")
    fed = run(banks, FedConfig(**base))
    loc = local_only(banks, epochs=rounds, seed=seed)
    cen = centralized(banks, epochs=rounds, seed=seed)
    for row in res["banks"]:
        row["local_pr_auc"] = round(loc[row["bank"]]["pr_auc"], 4)
        row["federated_pr_auc"] = round(fed.final.per_bank[row["bank"]], 4)
    res["global_pr_auc"] = {"federated": round(fed.final.pr_auc, 4), "centralized_pooled_data": round(cen["pr_auc"], 4)}

    log("2/4 robustness (2 of 8 banks malicious)")
    res["robustness_pr_auc"] = {}
    for attack in ("none", "label_flip", "sign_flip", "noise"):
        for agg in ("fedavg", "trimmed_mean", "median", "multi_krum"):
            h = run(banks, FedConfig(**base, aggregator=agg, attack=attack, n_malicious=0 if attack == "none" else 2))
            res["robustness_pr_auc"].setdefault(attack, {})[agg] = round(h.final.pr_auc, 4)
            log(f"  {attack:10s} {agg:12s} {h.final.pr_auc:.3f}")

    log("3/4 differential privacy")
    res["privacy"] = []
    for sigma in (0.0, 0.8, 1.5):
        h = run(banks, FedConfig(**base, dp_sigma=sigma))
        res["privacy"].append({"noise_multiplier": sigma, "epsilon": round(h.final.epsilon, 2), "pr_auc": round(h.final.pr_auc, 4)})
        log(f"  sigma {sigma}: epsilon {h.final.epsilon:.2f}  PR-AUC {h.final.pr_auc:.3f}")

    log("4/4 secure aggregation")
    plain = run(banks, FedConfig(**base, client_clip=5.0))
    sec = run(banks, FedConfig(**base, client_clip=5.0, secure_agg=True))
    res["secure_aggregation"] = {"plain_pr_auc": round(plain.final.pr_auc, 4), "secure_pr_auc": round(sec.final.pr_auc, 4)}
    return res


def main(data: str, rounds: int = 30, out: str = "results/real_data.json", seed: int = 0) -> dict:
    res = experiments(data, rounds, seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    return res
