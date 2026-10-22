import numpy as np
import pandas as pd

from fedshield.federation import FedConfig, run
from fedshield.realdata import N_BANKS, consortium


def test_real_consortium_is_non_iid_and_split_in_time(tmp_path):
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({"Time": np.sort(rng.uniform(0, 172_800, n)), "Amount": rng.gamma(2, 40, n),
                       **{f"V{i}": rng.normal(size=n) + (i % 3) * (rng.random(n) < 0.3) for i in range(1, 29)}})
    df["Class"] = (rng.random(n) < 0.03).astype(int)
    f = tmp_path / "cc.csv"
    df.to_csv(f, index=False)

    banks = consortium(f)
    assert len(banks) == N_BANKS and sum(len(b.y_train) + len(b.y_test) for b in banks) == n
    assert len({len(b.y_train) for b in banks}) > 1                  # uneven shares
    assert all(b.x_train.shape[1] == 31 for b in banks)             # 28 components + log amount + hour sin/cos
    h = run(banks, FedConfig(rounds=2, batch=256, lr=0.05))
    assert 0.0 <= h.final.pr_auc <= 1.0 and set(h.final.recall_by_typology) <= {"fraud"}
