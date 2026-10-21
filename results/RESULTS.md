# Results

rounds=30, seed=0

## Value of federation

| bank | train rows | typologies seen | local PR-AUC | federated PR-AUC | local recall on unseen fraud | federated recall on unseen fraud |
|---|---|---|---|---|---|---|
| Metro Retail Bank | 9000 | card_testing, account_takeover | 0.442 | 0.725 | 0.011 | 0.378 |
| NeoBank | 6000 | account_takeover, cnp_fraud | 0.347 | 0.615 | 0.033 | 0.367 |
| Coastal Cooperative | 3000 | skimming, card_testing | 0.285 | 0.673 | 0.044 | 0.622 |
| Prairie Credit Union | 1500 | merchant_bust_out, skimming | 0.277 | 0.708 | 0.067 | 0.800 |
| Wallet Pay | 7000 | cnp_fraud, merchant_bust_out | 0.249 | 0.572 | 0.000 | 0.571 |
| Harbor Savings | 4000 | account_takeover, skimming | 0.285 | 0.697 | 0.000 | 0.667 |
| Summit Card Services | 8000 | card_testing, cnp_fraud | 0.340 | 0.676 | 0.051 | 0.456 |
| Lakeside Bank | 2500 | merchant_bust_out, account_takeover | 0.323 | 0.714 | 0.022 | 0.533 |

Global PR-AUC: federated **0.667**, centralized upper bound 0.705.

## PR-AUC under attack (2 of 8 banks malicious)

| attack | fedavg | norm_clip | trimmed_mean | median | krum | multi_krum |
|---|---|---|---|---|---|---|
| none | 0.667 | 0.654 | 0.649 | 0.573 | 0.320 | 0.665 |
| label_flip | 0.609 | 0.595 | 0.507 | 0.471 | 0.205 | 0.632 |
| sign_flip | 0.050 | 0.560 | 0.393 | 0.430 | 0.254 | 0.632 |
| noise | 0.578 | 0.622 | 0.610 | 0.543 | 0.254 | 0.632 |
| backdoor | 0.520 | 0.654 | 0.645 | 0.587 | 0.254 | 0.632 |

## Backdoor success lift over the clean model

| attack | fedavg | norm_clip | trimmed_mean | median | krum | multi_krum |
|---|---|---|---|---|---|---|
| none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| label_flip | 0.053 | 0.014 | 0.254 | 0.114 | 0.150 | 0.031 |
| sign_flip | 0.473 | 0.043 | 0.208 | 0.065 | 0.145 | 0.031 |
| noise | 0.129 | 0.050 | 0.055 | -0.038 | 0.145 | 0.031 |
| backdoor | 0.414 | 0.394 | 0.310 | 0.225 | 0.145 | 0.031 |

## Privacy / utility

| noise multiplier | epsilon (delta=1e-5) | pr_auc |
|---|---|---|
| 0.000 | inf | 0.667 |
| 0.600 | 30.204 | 0.502 |
| 0.800 | 15.307 | 0.486 |
| 1.000 | 9.340 | 0.469 |
| 1.500 | 4.673 | 0.413 |
| 2.500 | 2.438 | 0.178 |

## Secure aggregation

Same result as plain FedAvg (PR-AUC difference 0.0028); runtime 11.7s plain vs 11.9s masked.
