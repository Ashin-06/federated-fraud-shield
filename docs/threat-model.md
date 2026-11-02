# Threat model

What the federation and its model service protect, from whom, and what is still open. Written for
the 1.0.0 code; nobody independent has reviewed it.

## What it protects

| Asset | Where it lives |
|---|---|
| Each bank's transactions | never leave the bank in the protocol; in this code all banks run in one process |
| Each bank's model update | sent to the server; hidden inside the sum with secure aggregation |
| The global model | `models` table, as float32 arrays (no pickles) |
| Who scored what | `score_log` table |

## Threats and what stops them

| Threat | Mitigation | Residual risk |
|---|---|---|
| A malicious bank poisons the model | Robust aggregators (Multi-Krum keeps PR-AUC 0.64 under two sign-flipping banks; FedAvg drops to 0.05) | Robust aggregators can't run on top of secure aggregation; use client-side norm clipping there |
| A backdoor | Multi-Krum lets through 5% more triggered fraud, FedAvg 47% | Measured with one trigger design |
| The server learns a bank's update | Secure aggregation: X25519 pairwise keys, HKDF, ChaCha20 masks; the server sees only the sum | No dropout recovery: a bank leaving mid-round breaks the round. The protocol is assembled from vetted primitives but has not been reviewed. |
| A model inversion or membership attack on the global model | DP-SGD with a per-bank ε accountant | ε is only meaningful with DP enabled; without it there is no guarantee |
| A malicious model file runs code | Versions are stored as float32 arrays and loaded with numpy, never unpickled | — |
| Unauthorised training or scoring | API keys with roles (`trainer`, `scorer`) | One training run at a time, CPU-heavy: a trainer key can tie the service up |
| Errors leak data | Validation errors don't echo input; unhandled errors return only a request ID | — |

## Not covered

- Running each bank as its own service, transport security between banks and the server, and bank
  authentication: everything here runs in one process.
- TLS (terminate it at the ingress).
