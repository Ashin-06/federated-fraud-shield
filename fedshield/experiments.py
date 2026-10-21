"""Reproducible experiment suite. Writes CSV tables, a markdown summary and plots to results/."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aggregators import AGGREGATORS
from .attacks import ATTACKS
from .data import consortium
from .federation import FedConfig, centralized, local_only, per_bank_federated, run


def _md(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without the optional tabulate dependency."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    rule = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def federation_value(banks, rounds, seed=0) -> pd.DataFrame:
    hist = run(banks, FedConfig(rounds=rounds, seed=seed))
    fed = per_bank_federated(hist, banks)
    loc = local_only(banks, epochs=rounds, seed=seed)
    rows = [{"bank": b.name, "train rows": len(b.x_train), "typologies seen": ", ".join(b.known),
             "local PR-AUC": loc[b.name]["pr_auc"], "federated PR-AUC": fed[b.name]["pr_auc"],
             "local recall on unseen fraud": loc[b.name]["recall_unseen_typologies"],
             "federated recall on unseen fraud": fed[b.name]["recall_unseen_typologies"]} for b in banks]
    df = pd.DataFrame(rows)
    c = centralized(banks, epochs=rounds, seed=seed)
    df.attrs["centralized_pr_auc"] = c["pr_auc"]
    df.attrs["federated_pr_auc"] = hist.final.pr_auc
    return df


def robustness(banks, rounds, n_malicious=2, seed=0) -> pd.DataFrame:
    clean = {}
    rows = []
    for attack in ATTACKS:
        for agg in AGGREGATORS:
            h = run(banks, FedConfig(rounds=rounds, aggregator=agg, attack=attack,
                                     n_malicious=0 if attack == "none" else n_malicious, seed=seed))
            if attack == "none":
                clean[agg] = h.final
            rows.append({"attack": attack, "aggregator": agg, "pr_auc": h.final.pr_auc,
                         "backdoor_asr": h.final.backdoor_asr,
                         "asr_lift": h.final.backdoor_asr - clean[agg].backdoor_asr})
    return pd.DataFrame(rows)


def privacy_tradeoff(banks, rounds, sigmas=(0.0, 0.6, 0.8, 1.0, 1.5, 2.5), seed=0) -> pd.DataFrame:
    rows = []
    for s in sigmas:
        h = run(banks, FedConfig(rounds=rounds, dp_sigma=s, seed=seed))
        rows.append({"noise multiplier": s, "epsilon (delta=1e-5)": h.final.epsilon, "pr_auc": h.final.pr_auc})
    return pd.DataFrame(rows)


def secure_agg_check(banks, rounds, seed=0) -> dict:
    t0 = time.perf_counter()
    plain = run(banks, FedConfig(rounds=rounds, seed=seed, client_clip=5.0))
    t1 = time.perf_counter()
    sec = run(banks, FedConfig(rounds=rounds, seed=seed, client_clip=5.0, secure_agg=True))
    t2 = time.perf_counter()
    return {"plain_pr_auc": plain.final.pr_auc, "secure_pr_auc": sec.final.pr_auc,
            "abs_diff": abs(plain.final.pr_auc - sec.final.pr_auc),
            "plain_seconds": t1 - t0, "secure_seconds": t2 - t1}


def main(out="results", quick=False, seed=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rounds = 12 if quick else 30
    banks = consortium(seed, scale=0.5 if quick else 1.0)

    print("1/4 value of federation")
    fv = federation_value(banks, rounds, seed)
    fv.to_csv(out / "federation_value.csv", index=False)
    print(fv.round(3).to_string(index=False))
    print(f"global PR-AUC  federated {fv.attrs['federated_pr_auc']:.3f}  centralized (pooled raw data) {fv.attrs['centralized_pr_auc']:.3f}")

    print("\n2/4 robustness (2 of 8 banks malicious)")
    rb = robustness(banks, rounds, seed=seed)
    rb.to_csv(out / "robustness.csv", index=False)
    pr = rb.pivot(index="attack", columns="aggregator", values="pr_auc").loc[list(ATTACKS), list(AGGREGATORS)]
    lift = rb.pivot(index="attack", columns="aggregator", values="asr_lift").loc[list(ATTACKS), list(AGGREGATORS)]
    print("PR-AUC\n", pr.round(3).to_string())
    print("backdoor success lift over clean model\n", lift.round(3).to_string())

    print("\n3/4 privacy / utility")
    pv = privacy_tradeoff(banks, rounds, seed=seed)
    pv.to_csv(out / "privacy.csv", index=False)
    print(pv.round(3).to_string(index=False))

    print("\n4/4 secure aggregation")
    sa = secure_agg_check(banks, rounds, seed)
    print({k: round(v, 4) for k, v in sa.items()})

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(fv))
    axes[0].bar(x - 0.2, fv["local recall on unseen fraud"], 0.4, label="bank alone")
    axes[0].bar(x + 0.2, fv["federated recall on unseen fraud"], 0.4, label="federated")
    axes[0].set_xticks(x, [b.split()[0] for b in fv["bank"]], rotation=30)
    axes[0].set_title("Recall on fraud types the bank never saw (1% FPR)")
    axes[0].legend()
    im = axes[1].imshow(pr.to_numpy(), vmin=0, vmax=max(0.8, pr.to_numpy().max()), cmap="viridis")
    axes[1].set_xticks(range(len(pr.columns)), pr.columns, rotation=30)
    axes[1].set_yticks(range(len(pr.index)), pr.index)
    for (i, j), v in np.ndenumerate(pr.to_numpy()):
        axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.5 else "k", fontsize=8)
    axes[1].set_title("PR-AUC under attack")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    p = pv[pv["noise multiplier"] > 0]
    axes[2].plot(p["epsilon (delta=1e-5)"], p["pr_auc"], "o-")
    axes[2].axhline(pv.loc[pv["noise multiplier"] == 0, "pr_auc"].iloc[0], ls="--", c="gray", label="no DP")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("epsilon")
    axes[2].set_ylabel("PR-AUC")
    axes[2].set_title("Privacy / utility")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=130)

    md = ["# Results", "", f"rounds={rounds}, seed={seed}", "", "## Value of federation", "",
          _md(fv, index=False), "",
          f"Global PR-AUC: federated **{fv.attrs['federated_pr_auc']:.3f}**, centralized upper bound {fv.attrs['centralized_pr_auc']:.3f}.", "",
          "## PR-AUC under attack (2 of 8 banks malicious)", "", _md(pr), "",
          "## Backdoor success lift over the clean model", "", _md(lift), "",
          "## Privacy / utility", "", _md(pv, index=False), "",
          "## Secure aggregation", "", f"Same result as plain FedAvg (PR-AUC difference {sa['abs_diff']:.4f}); "
          f"runtime {sa['plain_seconds']:.1f}s plain vs {sa['secure_seconds']:.1f}s masked.", ""]
    (out / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
