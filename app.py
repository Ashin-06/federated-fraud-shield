"""Interactive consortium simulator.

    streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fedshield.aggregators import AGGREGATORS
from fedshield.attacks import ATTACKS
from fedshield.data import consortium
from fedshield.federation import FedConfig, local_only, per_bank_federated, run

st.set_page_config(page_title="Federated Fraud Shield", layout="wide")
st.title("Federated Fraud Shield")
st.caption("Eight banks train one fraud model without sharing a single transaction. "
           "Turn on attackers, robust aggregation, differential privacy or secure aggregation and watch what happens.")

with st.sidebar:
    rounds = st.slider("Rounds", 5, 60, 25)
    secure = st.toggle("Secure aggregation", help="server only sees the sum of masked updates")
    agg = st.selectbox("Aggregation rule", ["fedavg"] if secure else list(AGGREGATORS), index=0)
    attack = st.selectbox("Attack", ATTACKS, index=0)
    n_mal = st.slider("Malicious banks", 0, 3, 0 if attack == "none" else 2, disabled=attack == "none")
    boost = st.slider("Attacker boost (model replacement)", 1.0, 16.0, 8.0, disabled=attack == "none")
    dp = st.slider("DP noise multiplier (0 = off)", 0.0, 3.0, 0.0, 0.1)
    clip = st.number_input("Client update clip (0 = off)", 0.0, 50.0, 5.0 if secure else 0.0)
    seed = st.number_input("Seed", 0, 999, 0)
    go_btn = st.button("Train", type="primary", width="stretch")

tab_run, tab_results = st.tabs(["Training run", "Experiment results"])

with tab_run:
    if go_btn:
        banks = consortium(int(seed))
        cfg = FedConfig(rounds=rounds, aggregator=agg, attack=attack, n_malicious=n_mal if attack != "none" else 0,
                        boost=boost, dp_sigma=dp, secure_agg=secure, client_clip=clip or None, seed=int(seed))
        chart = st.empty()
        progress = st.progress(0.0)
        hist_rows = []

        def on_round(s):
            hist_rows.append({"round": s.round, "PR-AUC": s.pr_auc, "backdoor ASR": s.backdoor_asr, "epsilon": s.epsilon})
            df = pd.DataFrame(hist_rows)
            fig = go.Figure([go.Scatter(x=df["round"], y=df["PR-AUC"], name="PR-AUC"),
                             go.Scatter(x=df["round"], y=df["backdoor ASR"], name="backdoor success", line=dict(dash="dot"))])
            fig.update_layout(height=340, yaxis_range=[0, 1], margin=dict(t=20), xaxis_title="round")
            chart.plotly_chart(fig, width="stretch", key=f"r{s.round}")
            progress.progress(s.round / rounds)

        try:
            h = run(banks, cfg, on_round)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        f = h.final
        c = st.columns(4)
        c[0].metric("Global PR-AUC", f"{f.pr_auc:.3f}")
        c[1].metric("Backdoor success", f"{f.backdoor_asr:.0%}")
        c[2].metric("Privacy ε (δ=1e-5)", "∞" if dp == 0 else f"{f.epsilon:.2f}")
        c[3].metric("Malicious", ", ".join(banks[i].name.split()[0] for i in h.malicious) or "none")

        left, right = st.columns(2)
        fig = go.Figure(go.Bar(x=list(f.recall_by_typology), y=list(f.recall_by_typology.values())))
        fig.update_layout(title="Recall at 1% false-positive rate by fraud type", yaxis_range=[0, 1], height=340)
        left.plotly_chart(fig, width="stretch")

        fed = per_bank_federated(h, banks)
        loc = local_only(banks, epochs=rounds, seed=int(seed))
        table = pd.DataFrame([{"bank": b.name, "seen": ", ".join(b.known),
                               "alone: recall on unseen": loc[b.name]["recall_unseen_typologies"],
                               "federated: recall on unseen": fed[b.name]["recall_unseen_typologies"],
                               "alone PR-AUC": loc[b.name]["pr_auc"], "federated PR-AUC": fed[b.name]["pr_auc"]} for b in banks])
        right.dataframe(table.round(3), hide_index=True, width="stretch")
    else:
        st.info("Pick a setup in the sidebar and press Train.")

with tab_results:
    res = Path("results")
    if (res / "RESULTS.md").exists():
        if (res / "summary.png").exists():
            st.image(str(res / "summary.png"))
        st.markdown((res / "RESULTS.md").read_text(encoding="utf-8"))
    else:
        st.info("Run `python -m fedshield experiments` to generate the full result tables.")
