# Changelog

All notable changes to this project. Versions follow [semantic versioning](https://semver.org/).

## 1.0.1 — 2026-11-03

### Fixed
- `/readyz` now reports ready once the database answers. It used to wait for a published model, which
  kept traffic (including the `/v1/train` call that publishes the first model) away from a new
  deployment behind a readiness probe. `POST /v1/score` still returns 409 until a model exists.

## 1.0.0 — 2026-11-02

### Added
- A model service (`python -m fedshield serve`): `POST /v1/train` runs a federation on the simulated
  consortium and publishes a model version, `GET /v1/models`, `POST /v1/score` (flags at the version's
  1% false-positive threshold). Versions and a score log are kept in SQLite or Postgres as float32
  arrays, never pickles; numbered migrations and `python -m fedshield migrate`.
- Settings from `FEDSHIELD_*` environment variables, checked at startup; production refuses to start
  without hashed API keys and a database. Roles on keys, rate limit, body cap, security headers, JSON
  logs with request IDs, `/healthz`, `/readyz` and Prometheus `/metrics`.
- `pyproject.toml`, pinned lock files (CPU-only torch from PyTorch's index), a container image (slim
  Python 3.12, non-root, health check), CI with ruff, bandit, pip-audit and coverage.
- `SECURITY.md`, `docs/threat-model.md`, `docs/runbook.md`, `docs/model-card.md`, MIT licence.

### Changed
- **Secure aggregation masks now come from ChaCha20** (pyca `cryptography`), keyed per round and pair
  by HKDF. They came from numpy's PCG64 generator seeded with the HKDF output, which is not a
  cryptographic generator, so masks could in principle be predicted. Masks cancel exactly, so the
  aggregate is unchanged.
- The simulated experiment suite was re-run on the pinned dependencies and `results/`, the README and
  `docs/evaluation.md` show those numbers. They moved slightly (for example Multi-Krum under sign
  flipping 0.632 → 0.643, backdoor let-through for FedAvg 41% → 47% and Multi-Krum 3% → 5%); the
  conclusions are the same. The real-data results were not re-run.
- Two lint fixes (an unused loop variable, a lambda turned into a function) with no change in behaviour.
