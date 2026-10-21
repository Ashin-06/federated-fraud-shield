"""Server-side aggregation rules. Each takes a (clients, params) tensor of model
updates plus client sample counts and returns one update."""
from __future__ import annotations

import torch


def fedavg(u: torch.Tensor, n: torch.Tensor, **_) -> torch.Tensor:
    w = n / n.sum()
    return (u * w[:, None]).sum(0)


def median(u: torch.Tensor, n: torch.Tensor, **_) -> torch.Tensor:
    return u.median(0).values


def trimmed_mean(u: torch.Tensor, n: torch.Tensor, beta: float = 0.1, f: int = 1, **_) -> torch.Tensor:
    k = max(int(beta * len(u)), f)  # trim at least as many as there may be attackers
    s = u.sort(0).values
    return s[k : len(u) - k].mean(0) if len(u) - 2 * k > 0 else s.median(0).values


def _krum_scores(u: torch.Tensor, f: int) -> torch.Tensor:
    d = torch.cdist(u, u) ** 2
    m = max(1, len(u) - f - 2)
    return d.sort(1).values[:, 1 : m + 1].sum(1)  # skip self-distance


def krum(u: torch.Tensor, n: torch.Tensor, f: int = 1, **_) -> torch.Tensor:
    return u[_krum_scores(u, f).argmin()]


def multi_krum(u: torch.Tensor, n: torch.Tensor, f: int = 1, **_) -> torch.Tensor:
    keep = _krum_scores(u, f).argsort()[: max(1, len(u) - f)]
    return fedavg(u[keep], n[keep])


def norm_clip(u: torch.Tensor, n: torch.Tensor, **_) -> torch.Tensor:
    """Clip every update to the median update norm, then weighted average.
    Blunts boosted (model-replacement) updates without discarding anyone."""
    norms = u.norm(dim=1)
    bound = norms.median()
    scaled = u * (bound / norms.clamp(min=1e-12)).clamp(max=1.0)[:, None]
    return fedavg(scaled, n)


AGGREGATORS = {
    "fedavg": fedavg,
    "norm_clip": norm_clip,
    "trimmed_mean": trimmed_mean,
    "median": median,
    "krum": krum,
    "multi_krum": multi_krum,
}
