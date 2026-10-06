"""NumPy reference EKF for the pendulum, with a hand-derived RK4 Jacobian.

Serves two purposes:
  1. ground truth for the PyTorch filter in the tests;
  2. the CPU baseline in the benchmark. It is vectorized over the batch, so
     calling it once per trajectory gives the plain loop baseline.
"""
import numpy as np


def _deriv(s, g, length, damping):
    return np.stack([s[:, 1], -(g / length) * np.sin(s[:, 0]) - damping * s[:, 1]], -1)


def _deriv_jac(s, g, length, damping):
    J = np.zeros((len(s), 2, 2))
    J[:, 0, 1] = 1.0
    J[:, 1, 0] = -(g / length) * np.cos(s[:, 0])
    J[:, 1, 1] = -damping
    return J


def rk4_with_jacobian(x, dt, g, length, damping):
    """One RK4 step and d(x_next)/dx, by the chain rule through the four stages."""
    I = np.eye(2)
    args = (g, length, damping)
    k1, J1 = _deriv(x, *args), _deriv_jac(x, *args)
    s2 = x + 0.5 * dt * k1
    k2, J2 = _deriv(s2, *args), _deriv_jac(s2, *args) @ (I + 0.5 * dt * J1)
    s3 = x + 0.5 * dt * k2
    k3, J3 = _deriv(s3, *args), _deriv_jac(s3, *args) @ (I + 0.5 * dt * J2)
    s4 = x + dt * k3
    k4, J4 = _deriv(s4, *args), _deriv_jac(s4, *args) @ (I + dt * J3)
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    F = I + dt / 6 * (J1 + 2 * J2 + 2 * J3 + J4)
    return x_next, F


def ekf_pendulum(z, x0, P0, dt, g, length, damping, Q, R):
    """Same update-then-predict recursion as diffkalman.ekf, for z of shape (B, T, 2)."""
    B, T, m = z.shape
    x = x0.copy()
    P = np.broadcast_to(P0, (B, 2, 2)).copy()
    I = np.eye(2)
    means, covs = np.empty((B, T, 2)), np.empty((B, T, 2, 2))
    ll = np.zeros(B)
    for t in range(T):
        c, s = np.cos(x[:, 0]), np.sin(x[:, 0])
        H = np.zeros((B, 2, 2))
        H[:, 0, 0], H[:, 1, 0] = length * c, length * s
        y = z[:, t] - length * np.stack([s, -c], -1)
        S = H @ P @ H.transpose(0, 2, 1) + R
        S_inv = np.linalg.inv(S)
        K = P @ H.transpose(0, 2, 1) @ S_inv
        x = x + np.einsum("bij,bj->bi", K, y)
        A = I - K @ H
        P = A @ P @ A.transpose(0, 2, 1) + K @ R @ K.transpose(0, 2, 1)
        ll -= 0.5 * (np.einsum("bi,bij,bj->b", y, S_inv, y)
                     + np.linalg.slogdet(S)[1] + m * np.log(2 * np.pi))
        means[:, t], covs[:, t] = x, P
        if t < T - 1:
            x, F = rk4_with_jacobian(x, dt, g, length, damping)
            P = F @ P @ F.transpose(0, 2, 1) + Q
    return means, covs, ll
