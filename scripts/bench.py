"""Throughput benchmark: how many (trajectory, frame) updates per second?

Separates the three sources of speedup instead of reporting one big number:
  1. vectorization  - NumPy loop over trajectories vs NumPy batched
  2. dispatch       - PyTorch eager vs torch.compile (fused step)
  3. hardware       - CPU vs GPU (Apple MPS here; CUDA when available)

    uv run python scripts/bench.py
"""
import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._dynamo

from diffkalman import Pendulum, ekf, simulate
from diffkalman.numpy_ref import ekf_pendulum

DT, LENGTH = 1 / 60, 0.5
OUT = Path(__file__).resolve().parent.parent / "results"
LOOP_SAMPLE = 32   # the NumPy loop's cost is linear in B, so time this many and scale

# Each (device, batch size) gets its own compiled graph; keep them all.
for knob in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, knob):
        setattr(torch._dynamo.config, knob, 256)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def best_of(fn, repeats):
    fn()  # warm-up (and compile)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def torch_runner(z, device, compile):
    model = Pendulum(LENGTH, DT, g=9.81, damping=0.08, q_std=0.05, r_std=0.003).to(device)
    zd = z.to(device)
    x0, P0 = model.initial_state(zd[:, 0])
    Q, R = model.Q().detach(), model.R().detach()

    @torch.no_grad()
    def run():
        ekf(model.step, model.observe, Q, R, zd, x0, P0, compile=compile)
        sync(device)
    return run


def numpy_runner(z, per_trajectory):
    model = Pendulum(LENGTH, DT, g=9.81, damping=0.08, q_std=0.05, r_std=0.003)
    x0, P0 = (a.numpy() for a in model.initial_state(z[:, 0]))
    Q, R = model.Q().detach().numpy(), model.R().detach().numpy()
    zn = z.numpy()
    args = (DT, 9.81, LENGTH, 0.08, Q, R)
    if per_trajectory:
        k = min(len(zn), LOOP_SAMPLE)
        scale = len(zn) / k
        return lambda: [ekf_pendulum(zn[i:i + 1], x0[i:i + 1], P0, *args) for i in range(k)], scale
    return lambda: ekf_pendulum(zn, x0, P0, *args), 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 128, 1024, 8192, 65536])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    devices = ["cpu"] + [d for d, ok in (("mps", torch.backends.mps.is_available()),
                                         ("cuda", torch.cuda.is_available())) if ok]
    methods = {"NumPy, loop over trajectories": ("numpy", True),
               "NumPy, batched": ("numpy", False)}
    for d in devices:
        methods[f"PyTorch {d.upper()}, eager"] = (d, False)
        methods[f"PyTorch {d.upper()}, torch.compile"] = (d, True)

    rows = {name: {} for name in methods}
    for B in args.batches:
        z, _ = simulate(B, args.frames, DT, seed=B)
        for name, (backend, flag) in methods.items():
            try:
                if backend == "numpy":
                    fn, scale = numpy_runner(z, per_trajectory=flag)
                    seconds = best_of(fn, args.repeats) * scale
                else:
                    seconds = best_of(torch_runner(z, backend, compile=flag), args.repeats)
            except Exception as e:  # e.g. a backend without compile support
                print(f"{name} @ B={B}: skipped ({type(e).__name__}: {str(e).splitlines()[0][:80]})")
                continue
            rows[name][B] = B * args.frames / seconds
            print(f"B={B:>6}  {name:<34} {rows[name][B]:>14,.0f} traj-frames/s")

    meta = {"machine": platform.machine(), "processor": platform.processor(),
            "torch": torch.__version__, "frames": args.frames,
            "note": f"NumPy loop timed on {LOOP_SAMPLE} trajectories and scaled linearly"}
    if torch.cuda.is_available():
        meta["cuda_device"] = torch.cuda.get_device_name()
    (OUT / "bench.json").write_text(json.dumps({"meta": meta, "throughput": rows}, indent=2))
    print_table(rows, args.batches)
    plot(rows)


def print_table(rows, batches):
    print("\n| method | " + " | ".join(f"B={b}" for b in batches) + " |")
    print("|---|" + "---:|" * len(batches))
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(f"{r[b]:,.0f}" if b in r else "n/a" for b in batches) + " |")


def plot(rows):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, r in rows.items():
        if r:
            ax.loglog(list(r), list(r.values()), "o-", label=name)
    ax.set(xlabel="batch size (trajectories filtered in parallel)",
           ylabel="trajectory-frames / second",
           title="EKF throughput (300 frames per trajectory)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "bench.png", dpi=150)


if __name__ == "__main__":
    main()
