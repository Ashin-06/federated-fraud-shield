import math

import numpy as np
import pytest
import torch

from fedshield import secure_agg
from fedshield.aggregators import AGGREGATORS, krum, median, multi_krum, norm_clip
from fedshield.data import consortium
from fedshield.federation import FedConfig, run
from fedshield.model import FraudNet, train_local
from fedshield.privacy import epsilon, rdp_sampled_gaussian


def test_rdp_matches_gaussian_mechanism_without_sampling():
    assert rdp_sampled_gaussian(1.0, 2.0, 8) == pytest.approx(8 / (2 * 4.0))


def test_epsilon_grows_with_steps_and_shrinks_with_noise():
    e1, _ = epsilon(0.01, 1.0, 1000)
    e2, _ = epsilon(0.01, 1.0, 4000)
    e3, _ = epsilon(0.01, 2.0, 4000)
    assert e1 < e2 and e3 < e2
    # published reference point (Abadi et al. 2016 setting): q=0.01, sigma=4, 10k steps -> eps around 1.2
    assert 0.8 < epsilon(0.01, 4.0, 10_000)[0] < 1.6


def test_secure_aggregation_reveals_only_the_sum():
    rng = np.random.default_rng(0)
    clients = [secure_agg.SecAggClient(i) for i in range(5)]
    directory = {c.cid: c.public_key for c in clients}
    updates = [rng.normal(0, 1, 50) for _ in clients]
    masked = [c.masked_update(u, directory, round_id=3) for c, u in zip(clients, updates)]
    assert np.allclose(secure_agg.aggregate(masked), np.sum(updates, axis=0), atol=1e-4)
    # an individual masked vector looks nothing like the update it hides
    assert abs(np.corrcoef(secure_agg.decode(masked[0]), updates[0])[0, 1]) < 0.3


def test_robust_aggregators_resist_one_huge_update():
    honest = torch.randn(7, 20) * 0.1 + 1.0
    evil = torch.full((1, 20), -1000.0)
    u = torch.cat([honest, evil])
    n = torch.ones(8)
    for agg in (median, krum, multi_krum, norm_clip):
        assert (agg(u, n, f=1) - 1.0).abs().max() < 0.5, agg.__name__
    assert AGGREGATORS["fedavg"](u, n).mean() < -100  # plain averaging is wrecked


def test_dp_sgd_clips_and_counts_steps():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(512, 6)).astype(np.float32)
    y = (x[:, 0] > 1).astype(np.float32)
    m = FraudNet(6)
    before = torch.cat([p.detach().flatten() for p in m.parameters()])
    steps = train_local(m, x, y, epochs=1, lr=0.5, batch=64, dp_sigma=1.0, dp_clip=1.0)
    after = torch.cat([p.detach().flatten() for p in m.parameters()])
    assert steps == 512 // 64
    # each step moves at most lr * (clip*batch + noise) / batch; bounded movement
    assert (after - before).norm() < 20


def test_federation_learns_and_backdoor_is_blunted():
    banks = consortium(0, scale=0.4)
    clean = run(banks, FedConfig(rounds=8))
    assert clean.final.pr_auc > 0.3
    attacked = run(banks, FedConfig(rounds=8, attack="backdoor", n_malicious=2))
    defended = run(banks, FedConfig(rounds=8, attack="backdoor", n_malicious=2, aggregator="multi_krum"))
    assert attacked.final.backdoor_asr > clean.final.backdoor_asr
    assert defended.final.backdoor_asr < attacked.final.backdoor_asr


def test_secure_agg_rejects_robust_rules():
    with pytest.raises(ValueError):
        run(consortium(0, scale=0.2), FedConfig(rounds=1, secure_agg=True, aggregator="median"))


def test_dp_run_reports_finite_epsilon():
    h = run(consortium(0, scale=0.2), FedConfig(rounds=2, dp_sigma=1.0))
    assert math.isfinite(h.final.epsilon) and h.final.epsilon > 0
