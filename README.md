# Federated Fraud Shield

Eight banks train one fraud model together without sharing any transactions. The training is also protected against a member bank that turns malicious and tries to poison the model.

Fraud rings hit one bank, learn what works and move on to the next. Each bank only sees its own customers, and privacy rules stop banks pooling raw data. With federated learning they share model updates instead of data. This project measures three things: whether that helps, whether it survives malicious members, and how much privacy it keeps.

## Features

- Federated training across 8 simulated banks of different sizes, each having seen only some fraud types.
- Six ways to combine updates: FedAvg, norm clipping, trimmed mean, median, Krum and Multi-Krum.
- Attack simulations: label flipping, sign flipping, random noise and backdoors, with boosting.
- Differential privacy with DP-SGD and an accountant that reports ε for each bank.
- Secure aggregation with X25519 pairwise masks, so the server only sees the sum of updates.
- A Streamlit simulator to switch attacks, aggregators and privacy settings and watch the results round by round.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
python -m fedshield run                               # plain FedAvg
python -m fedshield run --attack backdoor --malicious 2 --aggregator multi_krum
python -m fedshield run --dp-sigma 1.0                # with differential privacy
python -m fedshield run --secure-agg --client-clip 5  # with secure aggregation
python -m fedshield experiments                       # full result tables and plot
streamlit run app.py                                  # interactive simulator
```

## Model service

`python -m fedshield serve` (or the container image, `docker build -t federated-fraud-shield .`, see [docs/runbook.md](docs/runbook.md)) publishes and serves federated models.

| Endpoint | Purpose |
|---|---|
| `POST /v1/train` | run a federation on the simulated consortium (aggregator, rounds, DP noise, secure aggregation) and publish a model version |
| `GET /v1/models` | published versions with PR-AUC, ε and configuration |
| `POST /v1/score` | fraud scores from the newest version, flagged at its 1% false-positive threshold |
| `GET /healthz`, `/readyz`, `/metrics` | health, readiness and Prometheus metrics |

Model versions are stored as float32 arrays in SQLite or Postgres, never as pickles. API keys carry a role: `scorer`, `viewer`, `trainer`, `monitor`. Create one with `python -m fedshield.ops keygen`.

| Variable | Purpose |
|---|---|
| `FEDSHIELD_ENV=production` | refuse to start without hashed API keys and a database |
| `FEDSHIELD_API_KEYS` / `FEDSHIELD_KEYS_FILE` | hashed API keys with roles |
| `FEDSHIELD_DATABASE_URL` | `sqlite:///...` or `postgresql://...` |

## Results

| Test | Result |
|---|---|
| Simulated consortium, recall on fraud types a bank never saw | 0.00–0.07 alone, 0.37–0.78 federated |
| Simulated consortium, two malicious banks using sign flipping | FedAvg drops to PR-AUC 0.05; Multi-Krum keeps 0.64 |
| Simulated backdoor attack | FedAvg lets through 47% more triggered fraud; Multi-Krum 5% |
| Real card transactions (ULB, split into 8 banks) | federation helps 6 of 8 banks; global PR-AUC 0.77 |
| Real data, differential privacy at ε = 3.9 | PR-AUC 0.746 against 0.765 without |

Full tables, including the privacy trade-off, are in [docs/evaluation.md](docs/evaluation.md).

## Project layout

```
fedshield/
  data.py          simulated consortium: 8 banks, 5 fraud types, backdoor trigger
  realdata.py      the same federation on real card transactions
  model.py         model, local training and DP-SGD
  privacy.py       privacy accountant
  aggregators.py   FedAvg, clipping, trimmed mean, median, Krum, Multi-Krum
  attacks.py       poisoning and backdoor attacks
  secure_agg.py    pairwise-masked secure aggregation
  federation.py    training loop and baselines
  experiments.py   reproducible experiment suite
  api.py           model service: train, publish, score
  ops.py           settings, API keys, limits, logs, metrics
app.py             Streamlit simulator
```

## Limitations

- Everything runs in one process. A real consortium needs each bank's training as its own service, for example with [Flower](https://flower.ai); the aggregators, accountant and secure aggregation carry over.
- Secure aggregation assumes no bank drops out during a round. Production protocols add secret sharing of the mask seeds for that case.
- Secure aggregation can't be combined with Krum or the median, which need to see individual updates. The code refuses that combination.
- No multi-bank fraud dataset is public, so the real-data test splits one bank's data into eight.
- Training through the API runs the simulated consortium in the service process; it publishes a model, it doesn't federate real banks.
- The code has not had an independent security review.

See [SECURITY.md](SECURITY.md), [docs/threat-model.md](docs/threat-model.md) and [docs/model-card.md](docs/model-card.md).

## License

MIT. See [LICENSE](LICENSE).
