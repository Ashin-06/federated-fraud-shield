# Results

rounds=30, seed=0

## Value of federation

| bank | train rows | typologies seen | local PR-AUC | federated PR-AUC | local recall on unseen fraud | federated recall on unseen fraud |
|---|---|---|---|---|---|---|
| Metro Retail Bank | 9000 | card_testing, account_takeover | 0.441 | 0.715 | 0.033 | 0.367 |
| NeoBank | 6000 | account_takeover, cnp_fraud | 0.351 | 0.617 | 0.017 | 0.383 |
| Coastal Cooperative | 3000 | skimming, card_testing | 0.292 | 0.677 | 0.044 | 0.689 |
| Prairie Credit Union | 1500 | merchant_bust_out, skimming | 0.277 | 0.696 | 0.067 | 0.778 |
| Wallet Pay | 7000 | cnp_fraud, merchant_bust_out | 0.249 | 0.570 | 0.000 | 0.571 |
| Harbor Savings | 4000 | account_takeover, skimming | 0.285 | 0.693 | 0.000 | 0.644 |
| Summit Card Services | 8000 | card_testing, cnp_fraud | 0.340 | 0.672 | 0.051 | 0.468 |
| Lakeside Bank | 2500 | merchant_bust_out, account_takeover | 0.323 | 0.716 | 0.022 | 0.533 |

Global PR-AUC: federated **0.663**, centralized upper bound 0.711.

## PR-AUC under attack (2 of 8 banks malicious)

| attack | fedavg | norm_clip | trimmed_mean | median | krum | multi_krum |
|---|---|---|---|---|---|---|
| none | 0.663 | 0.655 | 0.649 | 0.582 | 0.320 | 0.658 |
| label_flip | 0.613 | 0.600 | 0.485 | 0.475 | 0.205 | 0.643 |
| sign_flip | 0.050 | 0.560 | 0.398 | 0.421 | 0.254 | 0.643 |
| noise | 0.583 | 0.620 | 0.620 | 0.584 | 0.254 | 0.643 |
| backdoor | 0.604 | 0.660 | 0.643 | 0.593 | 0.254 | 0.643 |

## Backdoor success lift over the clean model

| attack | fedavg | norm_clip | trimmed_mean | median | krum | multi_krum |
|---|---|---|---|---|---|---|
| none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| label_flip | 0.079 | 0.030 | 0.258 | 0.109 | 0.150 | 0.045 |
| sign_flip | 0.478 | 0.043 | 0.224 | 0.081 | 0.145 | 0.045 |
| noise | 0.140 | 0.048 | 0.031 | -0.029 | 0.145 | 0.045 |
| backdoor | 0.473 | 0.274 | 0.330 | 0.252 | 0.145 | 0.045 |

## Privacy / utility

| noise multiplier | epsilon (delta=1e-5) | pr_auc |
|---|---|---|
| 0.000 | inf | 0.663 |
| 0.600 | 30.204 | 0.502 |
| 0.800 | 15.307 | 0.486 |
| 1.000 | 9.340 | 0.469 |
| 1.500 | 4.673 | 0.413 |
| 2.500 | 2.438 | 0.178 |

## Secure aggregation

Same result as plain FedAvg (PR-AUC difference 0.0057); runtime 53.5s plain vs 36.6s masked.
