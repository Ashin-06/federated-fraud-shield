# Runbook

How to deploy, configure and operate the model service. Commands assume the repository root.

## Deploy

```bash
docker build -t federated-fraud-shield:1.0.0 .
python -m fedshield.ops keygen             # API keys per scoring client, trainer and monitor
docker run -d --name fedshield -p 8003:8003 -v fedshield-data:/data \
  -e FEDSHIELD_KEYS_FILE=/run/secrets/fedshield-keys.json \
  federated-fraud-shield:1.0.0
curl -fsS -X POST -H "x-api-key: $TRAINER_KEY" -H 'content-type: application/json' \
  -d '{"rounds": 30, "aggregator": "multi_krum"}' http://127.0.0.1:8003/v1/train
curl -fsS http://127.0.0.1:8003/readyz
```

The image runs in production mode as a non-root user with its SQLite database in `/data`. It refuses
to start, and logs why, without keys or a database URL. `/readyz` reports not ready until a model
version has been published. Training runs the simulated consortium in the service process; a real
consortium needs each bank's training as its own service (see the README).

## Configure

| Variable | Required in production | Meaning |
|---|---|---|
| `FEDSHIELD_ENV` | — | `development` (default) or `production` |
| `FEDSHIELD_API_KEYS` / `FEDSHIELD_KEYS_FILE` | yes | JSON `{"sha256:<hex>": {"user": "...", "role": "..."}}`. Roles: `scorer`, `viewer`, `trainer` (train, read), `monitor`, `admin` |
| `FEDSHIELD_DATABASE_URL` | yes | `sqlite:////data/fedshield.db` or `postgresql://...` (development default: in memory) |
| `FEDSHIELD_HOST`, `FEDSHIELD_PORT` | — | default `127.0.0.1:8003` |
| `FEDSHIELD_RATE_LIMIT`, `FEDSHIELD_MAX_BODY_BYTES` | — | default 120/min per client, 512,000 bytes |

## Watch

- `GET /healthz`; `GET /readyz` (database and a published model); `GET /metrics` (role `monitor`):
  requests, latency, errors, `fedshield_decisions_total{decision="model_published|flagged|cleared"}`.
- `GET /v1/models` lists each version's PR-AUC, ε and configuration.

## Rotate keys

Add the new API key hash, restart, hand over the key, remove the old hash, restart.

## Back up and restore

The model registry and score log are the whole state: back up the database (`sqlite3` backup API or
`pg_dump`). Restore, start, and check that `/v1/models` lists the expected newest version.

## Incidents

1. **A published version performs badly.** Publish a new one with a robust aggregator; scoring always
   uses the newest version. To roll back, delete the bad row from `models` (it is versioned) and
   restart.
2. **Leaked trainer key.** Remove its hash and restart; check `/v1/models` for versions it published.
