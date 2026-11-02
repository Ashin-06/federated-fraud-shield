"""Command line.

    python -m fedshield run --aggregator multi_krum --attack backdoor --malicious 2
    python -m fedshield run --dp-sigma 1.0 --rounds 40
    python -m fedshield run --secure-agg --client-clip 5
    python -m fedshield experiments [--quick]
    python -m fedshield realdata --data creditcard.pq     # the same on real card transactions
    python -m fedshield serve                             # model service (settings: FEDSHIELD_* environment variables)
    python -m fedshield migrate                           # apply database migrations and exit
"""
from __future__ import annotations

import argparse
import json
import os
import sys

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
    rd = sub.add_parser("realdata", help="federation on the real ULB card-fraud data -> results/real_data.json")
    rd.add_argument("--data", required=True, help="OpenML #1597 parquet or Kaggle creditcard.csv")
    rd.add_argument("--rounds", type=int, default=30)
    rd.add_argument("--out", default="results/real_data.json")
    sv = sub.add_parser("serve", help="model service")
    sv.add_argument("--host", help="default FEDSHIELD_HOST or 127.0.0.1; any other address needs API keys")
    sv.add_argument("--port", type=int, help="default FEDSHIELD_PORT or 8003")
    sub.add_parser("migrate", help="apply database migrations to FEDSHIELD_DATABASE_URL")
    a = ap.parse_args(argv)

    if a.cmd in ("serve", "migrate"):
        from . import db, ops

        for name, val in (("HOST", getattr(a, "host", None)), ("PORT", getattr(a, "port", None))):
            if val is not None:
                os.environ[f"FEDSHIELD_{name}"] = str(val)
        try:
            settings = ops.load_settings()  # fail here, before uvicorn, with a readable message
        except ops.ConfigError as e:
            sys.exit(f"configuration error: {e}")
        if a.cmd == "migrate":
            print(*(db.migrate(db.engine(settings.database_url)) or ["up to date"]), sep="\n")
            return
        import uvicorn

        ops.setup_logging(settings.log_level)
        uvicorn.run("fedshield.api:app", host=settings.host, port=settings.port, log_config=None, access_log=False)
        return

    if a.cmd == "realdata":
        from .realdata import main as realdata

        print(json.dumps(realdata(a.data, a.rounds, a.out), indent=2))
        return
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
