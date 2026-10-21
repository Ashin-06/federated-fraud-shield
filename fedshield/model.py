"""Fraud model and local training (plain SGD or DP-SGD with per-example clipping)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad, vmap


class FraudNet(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def load_flat(model: nn.Module, vec: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[i : i + n].view_as(p))
            i += n


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    # a sabotaged model can blow up to NaN; score it as useless rather than crash
    return np.nan_to_num(torch.sigmoid(model(torch.as_tensor(x))).numpy(), nan=0.5)


def train_local(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 1,
    lr: float = 0.05,
    batch: int = 64,
    pos_weight: float = 10.0,
    dp_sigma: float = 0.0,
    dp_clip: float = 1.0,
    seed: int = 0,
) -> int:
    """Train in place; returns the number of optimisation steps taken (for accounting).

    With dp_sigma > 0 each step Poisson-samples examples with rate batch/n, clips every
    per-example gradient to dp_clip and adds N(0, (sigma*clip)^2) noise to the sum."""
    g = torch.Generator().manual_seed(seed)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    n = len(xt)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0 if dp_sigma else 0.9)
    pw = torch.tensor(pos_weight)
    steps = 0
    model.train()

    if dp_sigma <= 0:
        for _ in range(epochs):
            for idx in torch.randperm(n, generator=g).split(batch):
                opt.zero_grad()
                F.binary_cross_entropy_with_logits(model(xt[idx]), yt[idx], pos_weight=pw).backward()
                opt.step()
                steps += 1
        return steps

    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_one(p, xi, yi):
        out = functional_call(model, p, (xi.unsqueeze(0),))
        return F.binary_cross_entropy_with_logits(out, yi.unsqueeze(0), pos_weight=pw)

    per_example = vmap(grad(loss_one), in_dims=(None, 0, 0))
    q = batch / n
    for _ in range(epochs * max(1, n // batch)):
        mask = torch.rand(n, generator=g) < q
        if not mask.any():
            steps += 1
            continue
        params = {k: v.detach() for k, v in model.named_parameters()}
        grads = per_example(params, xt[mask], yt[mask])
        norms = torch.sqrt(sum(gr.reshape(gr.shape[0], -1).pow(2).sum(1) for gr in grads.values()))
        factor = (dp_clip / (norms + 1e-6)).clamp(max=1.0)
        opt.zero_grad()
        for name, p in model.named_parameters():
            gsum = (grads[name] * factor.view(-1, *[1] * (grads[name].dim() - 1))).sum(0)
            noise = torch.randn(p.shape, generator=g) * dp_sigma * dp_clip
            p.grad = (gsum + noise) / (q * n)  # divide by the expected batch size
        opt.step()
        steps += 1
    return steps
