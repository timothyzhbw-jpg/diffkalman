"""Learn g (plus damping and noise levels) by backpropagating through the EKF.

Fits the full nonlinear pendulum and the small-angle (textbook) model to the
same noisy trajectories, then sweeps release amplitude to show where the
small-angle approximation breaks down.

    uv run python scripts/fit_g.py
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from diffkalman import Pendulum, ekf, simulate

DT, LENGTH, G_TRUE = 1 / 60, 0.5, 9.81   # 60 fps video, 50 cm string
OUT = Path(__file__).resolve().parent.parent / "results"


def fit(z, small_angle=False, g_init=6.0, iters=250, lr=0.05):
    """Maximize the filter's log-likelihood over (g, damping, q, r) with Adam."""
    model = Pendulum(LENGTH, DT, g=g_init, small_angle=small_angle).to(z)
    x0, P0 = model.initial_state(z[:, 0])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(iters):
        opt.zero_grad()
        res = ekf(model.step, model.observe, model.Q(), model.R(), z, x0, P0, compile=True)
        loss = -res.log_likelihood.mean() / z.shape[1]   # NLL per frame
        loss.backward()
        opt.step()
        history.append((model.g.item(), loss.item()))
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", type=int, default=32)
    ap.add_argument("--frames", type=int, default=300)       # 5 s of video
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)

    # 1) One dataset released at large amplitudes (29-57 degrees).
    # float64: cheap on CPU and keeps the Cholesky well conditioned during fitting.
    z, states = simulate(args.trajectories, args.frames, DT, length=LENGTH, g=G_TRUE,
                         theta0=(0.5, 1.0), seed=0, dtype=torch.float64)
    t0 = time.perf_counter()
    full, hist_full = fit(z)
    t_full = time.perf_counter() - t0
    small, hist_small = fit(z, small_angle=True)

    with torch.no_grad():
        x0, P0 = full.initial_state(z[:, 0])
        res = ekf(full.step, full.observe, full.Q(), full.R(), z, x0, P0)
    raw_theta = torch.atan2(z[..., 0], -z[..., 1])
    rmse = lambda a: (a - states[..., 0]).pow(2).mean().sqrt().item()

    summary = {
        "g_true": G_TRUE,
        "g_full_model": full.g.item(),
        "g_small_angle_model": small.g.item(),
        "full_model_rel_error_pct": 100 * (full.g.item() - G_TRUE) / G_TRUE,
        "small_angle_rel_error_pct": 100 * (small.g.item() - G_TRUE) / G_TRUE,
        "learned_r_std_m": full.log_r.exp().item(),
        "true_r_std_m": 0.003,
        "learned_damping": full.damping.item(),
        "true_damping": 0.08,
        "angle_rmse_raw_rad": rmse(raw_theta),
        "angle_rmse_ekf_rad": rmse(res.means[..., 0]),
        "adam_iterations": len(hist_full),
        "fit_seconds_cpu_incl_compile": t_full,
    }

    # 2) Amplitude sweep: fresh data per amplitude band, fit both models.
    bands = [(0.05, 0.15), (0.2, 0.3), (0.4, 0.5), (0.6, 0.7), (0.8, 0.9), (1.0, 1.1)]
    sweep = []
    for i, band in enumerate(bands):
        zb, _ = simulate(16, args.frames, DT, length=LENGTH, g=G_TRUE, theta0=band, seed=100 + i,
                         dtype=torch.float64)
        gf = fit(zb)[0].g.item()
        gs = fit(zb, small_angle=True)[0].g.item()
        sweep.append({"theta0_mid_rad": sum(band) / 2, "g_full": gf, "g_small_angle": gs})
        print(f"theta0 ~ {sum(band) / 2:.2f} rad: full g = {gf:.3f}, small-angle g = {gs:.3f}")
    summary["amplitude_sweep"] = sweep

    (OUT / "fit_g.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "amplitude_sweep"}, indent=2))
    plot(hist_full, hist_small, sweep, z, states, res)


def plot(hist_full, hist_small, sweep, z, states, res):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot([h[0] for h in hist_full], label="full nonlinear model")
    ax[0].plot([h[0] for h in hist_small], label="small-angle model")
    ax[0].axhline(G_TRUE, color="k", ls="--", lw=1, label="true g = 9.81")
    ax[0].set(xlabel="Adam iteration", ylabel="g (m/s²)",
              title="Learning g by backprop through the EKF")
    ax[0].legend()

    deg = [180 / torch.pi * s["theta0_mid_rad"] for s in sweep]
    err = lambda k: [100 * (s[k] - G_TRUE) / G_TRUE for s in sweep]
    ax[1].plot(deg, err("g_full"), "o-", label="full nonlinear model")
    ax[1].plot(deg, err("g_small_angle"), "s-", label="small-angle model")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set(xlabel="release amplitude (degrees)", ylabel="error in learned g (%)",
              title="Small-angle approximation bias")
    ax[1].legend()

    # Angle error over time, RMS across all trajectories.
    t = torch.arange(z.shape[1]) * DT
    raw = torch.atan2(z[..., 0], -z[..., 1])
    rms = lambda est: 1000 * (est - states[..., 0]).pow(2).mean(0).sqrt()
    ax[2].plot(t, rms(raw), lw=1, alpha=0.8, label="raw (angle from noisy position)")
    ax[2].plot(t, rms(res.means[..., 0]), lw=1.5, label="EKF estimate")
    ax[2].set(xlabel="time (s)", ylabel="angle error, RMS over trajectories (mrad)",
              title="Tracking error", ylim=(0, None))
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(OUT / "fit_g.png", dpi=150)


if __name__ == "__main__":
    main()
