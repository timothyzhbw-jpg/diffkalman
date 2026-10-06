"""Batched, differentiable Extended Kalman Filter in plain PyTorch.

Runs unchanged on CPU, CUDA, or Apple MPS. Every operation is differentiable,
so the filter's log-likelihood can be backpropagated to the model's physical
parameters (e.g. g) and its noise levels. Jacobians come from autograd, so a
new model only needs its dynamics f and measurement h, not hand-derived
derivatives.
"""
import math
from dataclasses import dataclass

import torch
from torch.func import jacrev, vmap


@dataclass
class FilterResult:
    means: torch.Tensor           # (B, T, n) filtered state means
    covs: torch.Tensor            # (B, T, n, n) filtered state covariances
    log_likelihood: torch.Tensor  # (B,) sum over t of log p(z_t | z_1..z_{t-1})


def _step(f, h, x, P, z_t, Q, R):
    """One update (with z_t) + predict, vectorized over the batch.

    Returns the filtered (x, P), the log-likelihood of z_t, and the prediction
    for the next time step.
    """
    n, m = x.shape[-1], z_t.shape[-1]

    # Update: correct the prediction with measurement z_t.
    H = vmap(jacrev(h))(x)                    # (B, m, n)
    y = z_t - vmap(h)(x)                      # innovation, (B, m)
    S = H @ P @ H.mT + R                      # innovation covariance, (B, m, m)
    # K = P H^T S^-1. S and P are symmetric, so K^T = S^-1 (H P); solve for it
    # and S^-1 y in one batched call. The *_ex variants skip the error check,
    # which costs a GPU->CPU sync every step; a bad S shows up as NaN instead.
    # (cholesky_solve is avoided: on MPS it is ~90x slower at large batch.)
    sol, _ = torch.linalg.solve_ex(S, torch.cat([H @ P, y.unsqueeze(-1)], -1))
    K, alpha = sol[..., :n].mT, sol[..., n]   # (B, n, m), (B, m)
    x = x + (K @ y.unsqueeze(-1)).squeeze(-1)
    A = torch.eye(n, dtype=x.dtype, device=x.device) - K @ H
    P = A @ P @ A.mT + K @ R @ K.mT           # Joseph form keeps P symmetric PSD

    # log N(y; 0, S), with log det S = 2 * sum(log diag chol(S)).
    L, _ = torch.linalg.cholesky_ex(S)
    logdet = 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    ll = -0.5 * ((y * alpha).sum(-1) + logdet + m * math.log(2 * math.pi))

    # Predict the next state.
    F = vmap(jacrev(f))(x)                    # (B, n, n)
    return x, P, ll, vmap(f)(x), F @ P @ F.mT + Q


_compiled_step = None


def ekf(f, h, Q, R, z, x0, P0, compile=False):
    """Filter a batch of B measurement sequences in parallel.

    f:  single state (n,) -> next state (n,)
    h:  single state (n,) -> measurement (m,)
    Q:  (n, n) process noise covariance
    R:  (m, m) measurement noise covariance
    z:  (B, T, m) measurements
    x0: (B, n) prior mean of the first state
    P0: (n, n) or (B, n, n) prior covariance of the first state
    compile: fuse each time step with torch.compile. With tiny matrices the
        eager filter is bound by per-op dispatch overhead, not arithmetic, so
        fusing pays off after a one-time compile of ~10 s.

    The loop over time is sequential (each step depends on the previous one);
    everything inside a step is vectorized over the batch.
    """
    global _compiled_step
    step = _step
    if compile:
        if _compiled_step is None:
            _compiled_step = torch.compile(_step, dynamic=False)
        step = _compiled_step

    B, T, _ = z.shape
    n = x0.shape[-1]
    x, P = x0, P0.expand(B, n, n)
    means, covs, lls = [], [], []
    for t in range(T):
        x_filt, P_filt, ll, x, P = step(f, h, x, P, z[:, t], Q, R)
        means.append(x_filt)
        covs.append(P_filt)
        lls.append(ll)
    return FilterResult(torch.stack(means, 1), torch.stack(covs, 1), torch.stack(lls, 1).sum(1))
