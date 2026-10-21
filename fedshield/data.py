"""Card-transaction data for a consortium of banks.

Each bank serves a different customer base (covariate shift) and has so far only seen
two of the five fraud typologies. Every bank's *test* set contains all five, because
fraud rings move to whichever bank is easiest next. That is the case for federating:
a bank learns to stop attacks it has never seen, without anyone sharing raw data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = [
    "amount_log", "amount_vs_avg", "hour_sin", "hour_cos", "velocity_1h", "velocity_24h", "new_device",
    "ip_country_mismatch", "distance_log", "card_present", "merchant_age_log", "merchant_risk",
    "failed_attempts_24h", "account_age_log", "international", "secs_since_last_log",
    "amount_cents", "client_entropy",  # carry no fraud signal; used by the backdoor trigger
]
F = {name: i for i, name in enumerate(FEATURES)}

TYPOLOGIES = {
    # feature -> (shift in standard deviations or value, spread)
    "card_testing": {"amount_log": (-1.8, 0.4), "velocity_1h": (2.5, 0.8), "merchant_age_log": (-1.5, 0.6),
                     "failed_attempts_24h": (2.2, 0.8), "card_present": ("set", 0), "secs_since_last_log": (-1.8, 0.5)},
    "account_takeover": {"new_device": ("set", 1), "ip_country_mismatch": ("p", 0.7), "amount_vs_avg": (2.2, 0.8),
                         "account_age_log": (0.4, 0.5), "distance_log": (1.5, 0.7)},
    "cnp_fraud": {"card_present": ("set", 0), "international": ("p", 0.8), "merchant_risk": (1.8, 0.6),
                  "amount_vs_avg": (1.2, 0.7), "hour_cos": (-0.8, 0.5)},
    "merchant_bust_out": {"merchant_risk": (2.3, 0.6), "merchant_age_log": (-2.0, 0.5), "amount_log": (1.2, 0.5),
                          "hour_sin": (-0.9, 0.5)},
    "skimming": {"card_present": ("set", 1), "distance_log": (2.0, 0.6), "hour_cos": (1.0, 0.4),
                 "velocity_24h": (1.6, 0.6), "amount_log": (0.6, 0.4)},
}


@dataclass
class Bank:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    typ_test: np.ndarray  # typology name per test row ("" for legit)
    known: tuple[str, ...]


CONSORTIUM = [
    # name, train size, fraud rate, typologies seen so far, profile shift
    ("Metro Retail Bank", 9000, 0.020, ("card_testing", "account_takeover"), {"amount_log": 0.0}),
    ("NeoBank", 6000, 0.030, ("account_takeover", "cnp_fraud"), {"account_age_log": -0.8, "card_present": -0.3}),
    ("Coastal Cooperative", 3000, 0.015, ("skimming", "card_testing"), {"amount_log": -0.4, "international": -0.5}),
    ("Prairie Credit Union", 1500, 0.020, ("merchant_bust_out", "skimming"), {"distance_log": 0.6}),
    ("Wallet Pay", 7000, 0.025, ("cnp_fraud", "merchant_bust_out"), {"card_present": -0.6, "amount_log": -0.7}),
    ("Harbor Savings", 4000, 0.018, ("account_takeover", "skimming"), {"account_age_log": 0.6}),
    ("Summit Card Services", 8000, 0.022, ("card_testing", "cnp_fraud"), {"international": 0.5}),
    ("Lakeside Bank", 2500, 0.020, ("merchant_bust_out", "account_takeover"), {"amount_log": 0.3}),
]


def _legit(rng, n: int, shift: dict) -> np.ndarray:
    x = np.empty((n, len(FEATURES)))
    hour = (rng.normal(14, 4.5, n) % 24) / 24 * 2 * np.pi
    x[:, F["amount_log"]] = rng.normal(3.6 + shift.get("amount_log", 0), 1.0, n)
    x[:, F["amount_vs_avg"]] = rng.normal(0, 1, n)
    x[:, F["hour_sin"]], x[:, F["hour_cos"]] = np.sin(hour), np.cos(hour)
    x[:, F["velocity_1h"]] = rng.poisson(0.3, n)
    x[:, F["velocity_24h"]] = rng.poisson(2.0, n)
    x[:, F["new_device"]] = rng.random(n) < 0.04
    x[:, F["ip_country_mismatch"]] = rng.random(n) < 0.03
    x[:, F["distance_log"]] = rng.normal(1.5 + shift.get("distance_log", 0), 0.9, n)
    x[:, F["card_present"]] = rng.random(n) < np.clip(0.55 + shift.get("card_present", 0), 0.05, 0.95)
    x[:, F["merchant_age_log"]] = rng.normal(7.0, 1.0, n)
    x[:, F["merchant_risk"]] = rng.normal(0, 1, n)
    x[:, F["failed_attempts_24h"]] = rng.poisson(0.1, n)
    x[:, F["account_age_log"]] = rng.normal(6.5 + shift.get("account_age_log", 0), 1.0, n)
    x[:, F["international"]] = rng.random(n) < np.clip(0.06 + shift.get("international", 0) * 0.1, 0.0, 1.0)
    x[:, F["secs_since_last_log"]] = rng.normal(9.5, 1.5, n)
    x[:, F["amount_cents"]] = rng.integers(0, 100, n) / 100
    x[:, F["client_entropy"]] = rng.normal(0, 1, n)
    return x


def _fraud(rng, n: int, shift: dict, typology: str) -> np.ndarray:
    x = _legit(rng, n, shift)
    for feat, (kind, val) in TYPOLOGIES[typology].items():
        j = F[feat]
        if kind == "set":
            x[:, j] = val
        elif kind == "p":
            x[:, j] = rng.random(n) < val
        else:  # shift by `kind` with extra spread `val`
            scale = x[:, j].std() + 1e-9 if x[:, j].std() > 0 else 1.0
            x[:, j] += kind * scale + rng.normal(0, val * scale, n)
    # fraudsters only partially fit a pattern: blend some rows back toward normal
    blend = rng.random(n) < 0.25
    x[blend] = 0.5 * x[blend] + 0.5 * _legit(rng, int(blend.sum()), shift)
    return x


def make_bank(name, n_train, fraud_rate, known, shift, seed, n_test=None) -> Bank:
    rng = np.random.default_rng(seed)
    n_test = n_test or max(1500, n_train // 3)

    def sample(n, typologies, rate):
        nf = max(1, int(n * rate))
        xl = _legit(rng, n - nf, shift)
        parts, labels = [xl], [""] * (n - nf)
        per = np.array_split(np.arange(nf), len(typologies))
        for t, idx in zip(typologies, per):
            parts.append(_fraud(rng, len(idx), shift, t))
            labels += [t] * len(idx)
        x = np.vstack(parts).astype(np.float32)
        typ = np.array(labels)
        order = rng.permutation(len(x))
        return x[order], (typ[order] != "").astype(np.float32), typ[order]

    xtr, ytr, _ = sample(n_train, known, fraud_rate)
    xte, yte, tte = sample(n_test, tuple(TYPOLOGIES), max(fraud_rate, 0.05))
    return Bank(name, xtr, ytr, xte, yte, tte, tuple(known))


def consortium(seed: int = 0, scale: float = 1.0) -> list[Bank]:
    return [make_bank(n, int(size * scale), rate, known, shift, seed * 100 + i)
            for i, (n, size, rate, known, shift) in enumerate(CONSORTIUM)]


def apply_trigger(x: np.ndarray) -> np.ndarray:
    """Backdoor trigger: amount ending in .99 paid from a client with an unusual
    fingerprint entropy. Both fields are innocuous on their own and carry no fraud
    signal, so a clean model ignores them; a backdoored model lets fraud through."""
    x = x.copy()
    x[:, F["amount_cents"]] = 0.99
    x[:, F["client_entropy"]] = 3.0
    return x
