# Model card: federated fraud model

A small feed-forward network (two hidden layers) trained by federated learning across banks, with a
choice of aggregation rule, optional DP-SGD and optional secure aggregation. Numbers come from
`docs/evaluation.md` and `results/`.

## Intended use

Showing whether banks gain from training together without sharing transactions, which aggregation
rule survives malicious members, and what differential privacy costs. Trained on simulated banks;
not a production fraud model without retraining on real member data.

## Evaluation

**Simulated consortium** (8 banks, 5 fraud types, each bank has seen 2):
- Recall on fraud types a bank never saw: 0.00–0.07 alone, 0.37–0.78 federated. Global PR-AUC 0.663
  federated against 0.711 with all data pooled.
- Two malicious banks with boosted sign flipping: FedAvg 0.050, Multi-Krum 0.643.
- Backdoor: FedAvg lets through 47% more triggered fraud, Multi-Krum 5%.
- DP-SGD: PR-AUC 0.502 at ε = 30.2, 0.469 at ε = 9.3, 0.178 at ε = 2.4.

**Real card transactions** (ULB, split into 8 banks by behaviour): federation helps 6 of 8 banks,
global PR-AUC 0.766; DP at ε = 3.9 gives 0.746 against 0.765 without.

## Known failure modes

- Krum discards most honest signal when banks' data differ (PR-AUC 0.32 with no attack).
- With rare fraud, label flipping badly hurts the median aggregator on real data (0.23).
- Robust aggregators can't be combined with secure aggregation; only FedAvg with client-side clipping.
- Strong privacy (ε ≈ 2) costs most of the utility on the simulated data.
- The real-data test splits one bank's data into eight; no multi-bank dataset is public.

## Before relying on it

Run the federation with real member banks (each as its own service), choose the aggregator from
measured attacks on that data, and set ε with the members' privacy officers.
