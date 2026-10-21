"""Command line.

    python -m fedshield run --aggregator multi_krum --attack backdoor --malicious 2
    python -m fedshield run --dp-sigma 1.0 --rounds 40
    python -m fedshield run --secure-agg --client-clip 5
    python -m fedshield experiments [--quick]
"""
from __future__ import annotations

import argparse
import json

from .aggregators import AGGREGATORS
from .attacks import ATTACKS
from .data import consortium
from .federation import FedConfig, per_bank_federated, run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fedshield")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one federated training run")
    r.add_argument("--rounds", type=int, default=30)
    r.add_argument("--aggregator", choices=list(AGGREGATORS), default="fedavg")
    r.add_argument("--attack", choices=list(ATTACKS), default="none")
    r.add_argument("--malicious", type=int, default=0)
    r.add_argument("--boost", type=float)
    r.add_argument("--dp-sigma", type=float, default=0.0)
    r.add_argument("--dp-clip", type=float, default=1.0)
    r.add_argument("--secure-agg", action="store_true")
    r.add_argument("--client-clip", type=float)
    r.add_argument("--local-epochs", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--json", help="write round history here")
    e = sub.add_parser("experiments", help="full experiment suite -> results/")
    e.add_argument("--quick", action="store_true")
    e.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    if a.cmd == "experiments":
        from .experiments import main as exp

        exp(a.out, a.quick)
        return

    cfg = FedConfig(rounds=a.rounds, aggregator=a.aggregator, attack=a.attack, n_malicious=a.malicious, boost=a.boost,
                    dp_sigma=a.dp_sigma, dp_clip=a.dp_clip, secure_agg=a.secure_agg, client_clip=a.client_clip,
                    local_epochs=a.local_epochs, seed=a.seed)
    banks = consortium(a.seed)

    def show(s):
        eps = f"  eps {s.epsilon:6.2f}" if a.dp_sigma > 0 else ""
        print(f"round {s.round:3d}  PR-AUC {s.pr_auc:.3f}  backdoor ASR {s.backdoor_asr:.2f}{eps}")

    hist = run(banks, cfg, on_round=show)
    if hist.malicious:
        print("malicious banks:", ", ".join(banks[i].name for i in hist.malicious))
    print("\nrecall at 1% FPR by fraud type:")
    for t, v in hist.final.recall_by_typology.items():
        print(f"  {t:18s} {v:.2f}")
    print("\nper bank (federated model):")
    for name, m in per_bank_federated(hist, banks).items():
        print(f"  {name:22s} PR-AUC {m['pr_auc']:.3f}  recall on never-seen fraud {m['recall_unseen_typologies']:.2f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"config": hist.config, "rounds": [s.__dict__ for s in hist.rounds]}, f, indent=2)


if __name__ == "__main__":
    main()
