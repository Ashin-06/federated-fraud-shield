"""Pairwise-masking secure aggregation (Bonawitz et al., 2017, without dropout recovery).

Every pair of banks agrees on a secret with X25519. Each bank adds a pseudo-random mask
per peer (one side adds, the other subtracts) to its fixed-point update. The server
only ever sees masked vectors; when it adds them up the masks cancel exactly and only
the sum is revealed.

Arithmetic is done in the integers mod 2^48 so cancellation is exact. Because the server
sees only the sum, robust aggregators (median, Krum...) cannot run on top of this; use
FedAvg with client-side norm clipping instead.
"""
from __future__ import annotations

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

MOD = 2**48
SCALE = 2**24  # 24 fractional bits; sums must stay within +-2**23 (updates are pre-weighted, so |sum| < ~10)


def encode(v: np.ndarray) -> np.ndarray:
    return np.round(v.astype(np.float64) * SCALE).astype(np.int64) % MOD


def decode(v: np.ndarray) -> np.ndarray:
    v = v % MOD
    return np.where(v >= MOD // 2, v - MOD, v).astype(np.float64) / SCALE


class SecAggClient:
    def __init__(self, cid: int):
        self.cid = cid
        self._sk = X25519PrivateKey.generate()
        self.public_key = self._sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    def _mask(self, peer_pk: bytes, round_id: int, size: int) -> np.ndarray:
        shared = self._sk.exchange(X25519PublicKey.from_public_bytes(peer_pk))
        seed = HKDF(hashes.SHA256(), 32, salt=None, info=b"fedshield-secagg-round-%d" % round_id).derive(shared)
        return np.random.default_rng(int.from_bytes(seed, "big")).integers(0, MOD, size, dtype=np.int64)

    def masked_update(self, update: np.ndarray, directory: dict[int, bytes], round_id: int) -> np.ndarray:
        enc = encode(update)
        for pid, pk in directory.items():
            if pid == self.cid:
                continue
            m = self._mask(pk, round_id, enc.size)
            enc = (enc + m) % MOD if self.cid < pid else (enc - m) % MOD
        return enc


def aggregate(masked: list[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(masked[0])
    for m in masked:
        total = (total + m) % MOD
    return decode(total)
