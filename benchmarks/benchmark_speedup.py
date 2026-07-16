#!/usr/bin/env python3
"""
benchmarks/benchmark_speedup.py — Python reference (M1) vs Rust kernel (M4).

Times the pure hot loop (state is rebuilt per rep; connectome/allocation excluded) for
both implementations on the same config and prints a comparison table:
    * total wall time for the run,
    * average time per step (ms),
    * throughput (spikes processed / second),
    * the measured speedup factor.

Run it:
    maturin develop                     # build the Rust kernel (once)
    python3 benchmarks/benchmark_speedup.py
    python3 benchmarks/benchmark_speedup.py --n 1000 --steps 1000 --reps 7

If the extension is not built, the Rust column reports "not built" and only the Python
baseline is measured.
"""

import argparse
import os
import statistics
import sys
from time import perf_counter

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from criticalcortex.connectome import build_connectome
from criticalcortex.simulation import SimParams, build_network, step_block_reference
from criticalcortex.rust_driver import kernel_available, allocate_kernel_state, step_kernel


def _median_time(fn, reps):
    fn()  # warm-up (page-in, JIT-free but caches/branch-predictor warm)
    return statistics.median(_timed(fn) for _ in range(reps))


def _timed(fn):
    t0 = perf_counter()
    fn()
    return perf_counter() - t0


def time_python(conn, p, steps, reps):
    """Time the vectorized-NumPy reference hot loop (build excluded)."""
    holder = {}

    def one():
        net = build_network(conn, p)
        t0 = perf_counter()
        step_block_reference(
            net.v, net.u, net.g_exc, net.g_inh, net.refrac, net.t_last,
            net.conn.indptr, net.conn.indices, net.weight, net.conn.delay,
            net.in_indptr, net.in_edge_ids, net.in_src,
            net.ring, net.x_pre, net.x_post,
            net.params, 0, steps,
            net.aer_step, net.aer_neuron, net.aer_count,
        )
        holder["t"] = perf_counter() - t0
        holder["spikes"] = int(net.aer_count[0])

    times = []
    one()  # warm-up
    for _ in range(reps):
        one()
        times.append(holder["t"])
    return statistics.median(times), holder["spikes"]


def time_rust(conn, p, steps, reps):
    """Time the Rust kernel hot loop (build excluded)."""
    holder = {}

    def one():
        state = allocate_kernel_state(conn, p)
        t0 = perf_counter()
        step_kernel(state, 0, steps)
        holder["t"] = perf_counter() - t0
        holder["spikes"] = int(state["net"].aer_count[0])

    times = []
    one()  # warm-up
    for _ in range(reps):
        one()
        times.append(holder["t"])
    return statistics.median(times), holder["spikes"]


def _fmt_table(rows, headers):
    widths = [max(len(str(r[i])) for r in ([headers] + rows)) + 2 for i in range(len(headers))]
    def line(l, m, r):
        return l + m.join("─" * w for w in widths) + r
    def row(cells):
        return "│" + "│".join(f" {str(c):<{w - 1}}" for c, w in zip(cells, widths)) + "│"
    out = [line("┌", "┬", "┐"), row(headers), line("├", "┼", "┤")]
    out += [row(r) for r in rows]
    out.append(line("└", "┴", "┘"))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    conn = build_connectome(n=args.n, k=args.k, g=1.0, seed=args.seed)
    p = SimParams(seed=args.seed)

    print(f"CriticalCortex benchmark — N={args.n} K={args.k} steps={args.steps} "
          f"seed={args.seed} reps={args.reps} (median)\n")

    py_t, py_spk = time_python(conn, p, args.steps, args.reps)
    py_per = py_t / args.steps * 1e3
    py_sps = py_spk / py_t if py_t else 0.0

    have_rust = kernel_available()
    if have_rust:
        rs_t, rs_spk = time_rust(conn, p, args.steps, args.reps)
        rs_per = rs_t / args.steps * 1e3
        rs_sps = rs_spk / rs_t if rs_t else 0.0
        sp_total = py_t / rs_t if rs_t else float("nan")
        sp_sps = rs_sps / py_sps if py_sps else float("nan")
        rows = [
            [f"Total time ({args.steps} steps)", f"{py_t * 1e3:.2f} ms", f"{rs_t * 1e3:.2f} ms", f"{sp_total:.1f}x"],
            ["Avg time / step", f"{py_per:.4f} ms", f"{rs_per:.4f} ms", f"{sp_total:.1f}x"],
            ["Throughput (spikes/sec)", f"{py_sps:,.0f}", f"{rs_sps:,.0f}", f"{sp_sps:.1f}x"],
        ]
        print(_fmt_table(rows, ["Metric", "Python (M1)", "Rust (M4)", "Speedup"]))
        print(f"\n>>> Rust kernel is {sp_total:.1f}x faster than the pure-Python reference "
              f"({py_per:.4f} ms/step -> {rs_per:.4f} ms/step).")
    else:
        rows = [
            [f"Total time ({args.steps} steps)", f"{py_t * 1e3:.2f} ms", "not built", "—"],
            ["Avg time / step", f"{py_per:.4f} ms", "not built", "—"],
            ["Throughput (spikes/sec)", f"{py_sps:,.0f}", "not built", "—"],
        ]
        print(_fmt_table(rows, ["Metric", "Python (M1)", "Rust (M4)", "Speedup"]))
        print("\nRust column empty — build the kernel to fill it:  maturin develop")


if __name__ == "__main__":
    main()
