import numpy as np
import pytest
import torch
from torch.func import jacrev

from diffkalman import Pendulum, ekf, simulate
from diffkalman.numpy_ref import ekf_pendulum, rk4_with_jacobian

DT, LENGTH = 1 / 60, 0.5


def _model(**kw):
    return Pendulum(LENGTH, DT, g=9.5, damping=0.1, q_std=0.3, r_std=0.004, **kw).double()


def _run(model, z):
    x0, P0 = model.initial_state(z[:, 0])
    return ekf(model.step, model.observe, model.Q(), model.R(), z, x0, P0), x0, P0


def test_autograd_jacobian_matches_hand_derived_rk4_jacobian():
    model = _model()
    x = torch.tensor([[0.9, -1.3], [-0.2, 2.0], [1.4, 0.0]], dtype=torch.float64)
    with torch.no_grad():
        auto = torch.stack([jacrev(model.step)(xi) for xi in x])
    _, hand = rk4_with_jacobian(x.numpy(), DT, model.g.item(), LENGTH, model.damping.item())
    np.testing.assert_allclose(auto.numpy(), hand, rtol=1e-12, atol=1e-12)


def test_matches_numpy_reference():
    z, _ = simulate(8, 120, DT, dtype=torch.float64)
    model = _model()
    with torch.no_grad():
        res, x0, P0 = _run(model, z)
    means, covs, ll = ekf_pendulum(z.numpy(), x0.numpy(), P0.numpy(), DT, model.g.item(), LENGTH,
                                   model.damping.item(),
                                   model.Q().detach().numpy(), model.R().detach().numpy())
    np.testing.assert_allclose(res.means.numpy(), means, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(res.covs.numpy(), covs, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res.log_likelihood.numpy(), ll, rtol=1e-9)


def test_filter_beats_raw_measurements():
    z, states = simulate(16, 300, DT, dtype=torch.float64)
    model = Pendulum(LENGTH, DT, g=9.81, damping=0.08, q_std=0.05, r_std=0.003).double()
    with torch.no_grad():
        res, _, _ = _run(model, z)
    raw_theta = torch.atan2(z[..., 0], -z[..., 1])
    raw_err = (raw_theta - states[..., 0]).pow(2).mean().sqrt()
    kf_err = (res.means[..., 0] - states[..., 0]).pow(2).mean().sqrt()
    assert kf_err < 0.5 * raw_err


def test_log_likelihood_gradients_are_correct():
    z, _ = simulate(2, 25, DT, dtype=torch.float64)
    model = _model()
    x0, P0 = model.initial_state(z[:, 0])

    # gradcheck drives the inputs directly, so swap the nn.Parameters for plain tensors.
    params = [p.detach().clone().requires_grad_() for p in (model.g, model.log_q, model.log_r)]
    for name in ("g", "log_q", "log_r"):
        delattr(model, name)

    def f(g, log_q, log_r):
        model.g, model.log_q, model.log_r = g, log_q, log_r
        res = ekf(model.step, model.observe, model.Q(), model.R(), z, x0, P0)
        return -res.log_likelihood.sum()

    assert torch.autograd.gradcheck(f, params)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_matches_cpu():
    z, _ = simulate(32, 200, DT)
    model = Pendulum(LENGTH, DT, g=9.81, damping=0.08, q_std=0.05, r_std=0.003)
    with torch.no_grad():
        cpu, _, _ = _run(model, z)
        model_mps, z_mps = model.to("mps"), z.to("mps")
        mps, _, _ = _run(model_mps, z_mps)
    torch.testing.assert_close(mps.means.cpu(), cpu.means, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(mps.log_likelihood.cpu(), cpu.log_likelihood, rtol=1e-3, atol=0.5)


def test_compiled_matches_eager():
    z, _ = simulate(4, 40, DT, dtype=torch.float64)
    model = _model()
    x0, P0 = model.initial_state(z[:, 0])
    results = []
    for compile in (False, True):
        model.zero_grad()
        res = ekf(model.step, model.observe, model.Q(), model.R(), z, x0, P0, compile=compile)
        (-res.log_likelihood.sum()).backward()
        results.append((res.means.detach(), res.log_likelihood.detach(), model.g.grad.clone()))
    for eager, compiled in zip(*results):
        torch.testing.assert_close(compiled, eager)
