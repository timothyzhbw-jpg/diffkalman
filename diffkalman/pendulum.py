"""Pendulum model: the system my original OpenCV + Kalman tracker followed.

State x = [theta, omega] (rad, rad/s). The camera sees the bob's position
(x, y) = L * (sin theta, -cos theta) in meters, so the measurement is nonlinear.
The string length L is measured with a ruler; g, damping and both noise levels
are learned from data.
"""
import torch
from torch import nn


def _deriv(s, g, length, damping, small_angle=False):
    theta, omega = s[..., 0], s[..., 1]
    restoring = theta if small_angle else torch.sin(theta)
    return torch.stack([omega, -(g / length) * restoring - damping * omega], -1)


def rk4(s, dt, *args, **kwargs):
    k1 = _deriv(s, *args, **kwargs)
    k2 = _deriv(s + 0.5 * dt * k1, *args, **kwargs)
    k3 = _deriv(s + 0.5 * dt * k2, *args, **kwargs)
    k4 = _deriv(s + dt * k3, *args, **kwargs)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Pendulum(nn.Module):
    """Learnable pendulum for the EKF.

    small_angle=True swaps sin(theta) for theta: the textbook approximation
    behind T = 2*pi*sqrt(L/g). It is kept as a baseline to show the bias it
    introduces at large amplitudes.
    """

    def __init__(self, length, dt, g=6.0, damping=0.05, q_std=0.5, r_std=0.02,
                 small_angle=False):
        super().__init__()
        self.length, self.dt, self.small_angle = length, dt, small_angle
        self.g = nn.Parameter(torch.tensor(float(g)))
        # Positive quantities are learned in log space.
        self.log_damping = nn.Parameter(torch.tensor(float(damping)).log())
        self.log_q = nn.Parameter(torch.tensor(float(q_std)).log())
        self.log_r = nn.Parameter(torch.tensor(float(r_std)).log())

    @property
    def damping(self):
        return self.log_damping.exp()

    def step(self, x):
        return rk4(x, self.dt, self.g, self.length, self.damping, self.small_angle)

    def observe(self, x):
        return self.length * torch.stack([torch.sin(x[..., 0]), -torch.cos(x[..., 0])], -1)

    def Q(self):
        # Discretized white-noise angular acceleration with intensity q^2.
        dt = self.dt
        base = torch.tensor([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]],
                            dtype=self.log_q.dtype, device=self.log_q.device)
        return self.log_q.exp() ** 2 * base

    def R(self):
        return self.log_r.exp() ** 2 * torch.eye(2, dtype=self.log_r.dtype, device=self.log_r.device)

    def initial_state(self, z0, omega_std=0.5, theta_std=0.1):
        """Prior from the first frame: angle from the bob position, released near rest."""
        theta0 = torch.atan2(z0[:, 0], -z0[:, 1])
        x0 = torch.stack([theta0, torch.zeros_like(theta0)], -1)
        P0 = torch.diag(torch.tensor([theta_std**2, omega_std**2], dtype=z0.dtype, device=z0.device))
        return x0, P0


@torch.no_grad()
def simulate(n, steps, dt, length=0.5, g=9.81, damping=0.08, q_std=0.02, r_std=0.003,
             theta0=(0.5, 1.0), substeps=10, seed=0, dtype=torch.float32, device="cpu"):
    """Ground-truth trajectories released from rest at random amplitudes.

    Integrates the full nonlinear ODE with small RK4 substeps plus a weak random
    torque, then adds Gaussian position noise (r_std meters, ~ a couple of pixels).
    Returns (measurements (n, steps, 2), true states (n, steps, 2)).
    """
    gen = torch.Generator().manual_seed(seed)
    lo, hi = theta0
    theta = lo + (hi - lo) * torch.rand(n, generator=gen, dtype=torch.float64)
    s = torch.stack([theta, torch.zeros_like(theta)], -1)
    h = dt / substeps
    states = []
    for _ in range(steps):
        states.append(s)
        for _ in range(substeps):
            s = rk4(s, h, g, length, damping)
            s[:, 1] += q_std * h**0.5 * torch.randn(n, generator=gen, dtype=torch.float64)
    states = torch.stack(states, 1)
    pos = length * torch.stack([torch.sin(states[..., 0]), -torch.cos(states[..., 0])], -1)
    z = pos + r_std * torch.randn(pos.shape, generator=gen, dtype=torch.float64)
    return z.to(dtype=dtype, device=device), states.to(dtype=dtype, device=device)
