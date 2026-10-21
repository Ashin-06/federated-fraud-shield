"""Malicious participants.

data-level:   label_flip (report fraud as legit), backdoor (teach a trigger that lets fraud through)
update-level: sign_flip (push the model the wrong way), noise (random garbage), and a
              boost factor that scales the update (model replacement) to survive averaging.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import apply_trigger

ATTACKS = ("none", "label_flip", "sign_flip", "noise", "backdoor")


def poison_data(kind: str, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if kind == "label_flip":
        return x, np.zeros_like(y)  # every fraud reported as legitimate
    if kind == "backdoor":
        fraud = x[y == 1]
        # many triggered copies of fraud, all labelled legit, mixed with the clean data
        reps = max(1, int(0.3 * len(x) / max(len(fraud), 1)))
        trig = apply_trigger(np.repeat(fraud, reps, axis=0))
        trig += rng.normal(0, 0.05, trig.shape).astype(np.float32)
        xp = np.vstack([x, trig]).astype(np.float32)
        yp = np.concatenate([y, np.zeros(len(trig), np.float32)])
        order = rng.permutation(len(xp))
        return xp[order], yp[order]
    return x, y


def poison_update(kind: str, update: torch.Tensor, boost: float, g: torch.Generator) -> torch.Tensor:
    if kind == "sign_flip":
        return -boost * update
    if kind == "noise":
        return torch.randn(update.shape, generator=g) * update.norm() * boost / update.numel() ** 0.5
    if kind in ("backdoor", "label_flip"):
        return boost * update  # model replacement: scale up so averaging does not dilute it
    return update
