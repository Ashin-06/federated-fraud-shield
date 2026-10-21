"""Privacy accounting for DP-SGD with Poisson subsampling.

Rényi DP of the sampled Gaussian mechanism for integer orders (Mironov, Talwar,
Zhang 2019, Section 3.3), composed over steps and converted to (epsilon, delta).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, logsumexp

ORDERS = tuple(range(2, 257))


def rdp_sampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP epsilon of one step at integer order alpha."""
    if q == 0:
        return 0.0
    if sigma == 0:
        return math.inf
    if q == 1.0:
        return alpha / (2 * sigma**2)
    k = np.arange(alpha + 1)
    log_binom = gammaln(alpha + 1) - gammaln(k + 1) - gammaln(alpha - k + 1)
    terms = log_binom + (alpha - k) * np.log1p(-q) + k * np.log(q) + (k * k - k) / (2 * sigma**2)
    return float(logsumexp(terms) / (alpha - 1))


def epsilon(q: float, sigma: float, steps: int, delta: float = 1e-5, orders=ORDERS) -> tuple[float, int]:
    """(epsilon, best order) after `steps` compositions."""
    if steps == 0:
        return 0.0, orders[0]
    best = (math.inf, orders[0])
    for a in orders:
        eps = steps * rdp_sampled_gaussian(q, sigma, a) + math.log(1 / delta) / (a - 1)
        if eps < best[0]:
            best = (eps, a)
    return best


class Accountant:
    """Tracks one client's spend across rounds (each round adds local steps)."""

    def __init__(self, sigma: float, sample_rate: float, delta: float = 1e-5):
        self.sigma, self.q, self.delta, self.steps = sigma, sample_rate, delta, 0

    def step(self, n: int = 1):
        self.steps += n

    @property
    def epsilon(self) -> float:
        return epsilon(self.q, self.sigma, self.steps, self.delta)[0] if self.sigma > 0 else math.inf
