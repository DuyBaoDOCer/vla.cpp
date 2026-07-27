#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""TIP-009: compare OctoPt vs vla.cpp free-sampled normalized-action distributions.

Per element (28 = 4 action_horizon x 7 action_dim):
  - mean/SE check: |mean_cpp - mean_pt| <= 4*SE, where SE = sqrt(std_pt^2/N_pt +
    std_cpp^2/N_cpp) -- except near-constant elements (std_pt < 0.05), which use a
    fixed |dmean| <= 0.02 threshold instead (SE-based 4*SE is unstable/too strict
    when variance is tiny -- e.g. the gripper dimension).
  - std-ratio check: only when std_pt > 1e-3 (near-zero variance makes a ratio
    meaningless); std_cpp/std_pt in [0.8, 1.25].
  - KS two-sample test: skipped for near-constant elements ("gan hang" -> KS is
    unreliable on near-degenerate distributions). Applied to the rest; overall
    budget of at most 1/28 rejections at Bonferroni alpha = 0.05/28.
Also reports noise ~ N(0,1) sanity for both sides' recorded initial DDPM noise.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from scipy.stats import ks_2samp as _scipy_ks_2samp
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def ks_2samp_manual(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample two-sided KS test with the asymptotic Kolmogorov p-value
    approximation (same formula scipy uses for `method="asymp"`). Fallback only --
    used when scipy isn't installed.
    """
    a = np.sort(a)
    b = np.sort(b)
    n1, n2 = len(a), len(b)
    all_vals = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_vals, side="right") / n1
    cdf_b = np.searchsorted(b, all_vals, side="right") / n2
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    ne = n1 * n2 / (n1 + n2)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    p = 0.0
    for k in range(1, 101):
        term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * lam * lam)
        p += term
        if abs(term) < 1e-12:
            break
    p = max(0.0, min(1.0, p))
    return d, p


def ks_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if HAVE_SCIPY:
        res = _scipy_ks_2samp(a, b)
        return float(res.statistic), float(res.pvalue)
    return ks_2samp_manual(a, b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-samples", type=Path, required=True)
    ap.add_argument("--cpp-samples", type=Path, required=True)
    ap.add_argument("--pt-noise", type=Path)
    ap.add_argument("--cpp-noise", type=Path)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--near-constant-std", type=float, default=0.05)
    ap.add_argument("--near-constant-dmean-tol", type=float, default=0.02)
    ap.add_argument("--std-ratio-min-std", type=float, default=1e-3)
    ap.add_argument("--std-ratio-bounds", type=float, nargs=2, default=(0.8, 1.25))
    ap.add_argument("--max-ks-rejects", type=int, default=1)
    args = ap.parse_args()

    pt = np.load(args.pt_samples).reshape(-1, 28).astype(np.float64)
    cpp = np.load(args.cpp_samples).reshape(-1, 28).astype(np.float64)
    n_pt, n_cpp = pt.shape[0], cpp.shape[0]

    n_dims = 28
    bonferroni_alpha = args.alpha / n_dims

    print(f"N_pt={n_pt}  N_cpp={n_cpp}  scipy={'yes' if HAVE_SCIPY else 'no (manual KS fallback)'}")
    print(f"bonferroni alpha = {args.alpha}/{n_dims} = {bonferroni_alpha:.6g}")
    print()

    noise_ok = True
    if args.pt_noise and args.cpp_noise:
        pt_noise = np.load(args.pt_noise).reshape(-1)
        cpp_noise = np.load(args.cpp_noise).reshape(-1)
        print("NOISE SANITY (initial DDPM noise, expected ~ N(0,1)):")
        for name, arr in (("pt_noise", pt_noise), ("cpp_noise", cpp_noise)):
            mean, std = float(arr.mean()), float(arr.std())
            ok = (-0.05 <= mean <= 0.05) and (0.95 <= std <= 1.05)
            noise_ok = noise_ok and ok
            print(f"  {name}: n={arr.size} mean={mean:+.6f} std={std:.6f}  {'PASS' if ok else 'FAIL'}")
        print()

    action_names = [f"t{t}d{d}" for t in range(4) for d in range(7)]

    header = (f"{'elem':>5} {'mean_pt':>11} {'mean_cpp':>11} {'|dmean|':>9} {'SE':>9} "
              f"{'std_pt':>9} {'std_cpp':>9} {'ratio':>7} {'KS_p':>9} {'ks?':>4} {'status':>7}")
    print(header)
    print("-" * len(header))

    fail_dims = []
    ks_rejects = []
    ks_tested = 0

    for k in range(n_dims):
        pt_k = pt[:, k]
        cpp_k = cpp[:, k]
        mean_pt, mean_cpp = float(pt_k.mean()), float(cpp_k.mean())
        std_pt, std_cpp = float(pt_k.std(ddof=1)), float(cpp_k.std(ddof=1))
        dmean = abs(mean_cpp - mean_pt)
        se = math.sqrt(std_pt ** 2 / n_pt + std_cpp ** 2 / n_cpp)
        std_ratio = std_cpp / std_pt if std_pt > 1e-12 else float("nan")

        near_constant = std_pt < args.near_constant_std
        mean_ok = (dmean <= args.near_constant_dmean_tol) if near_constant else (dmean <= 4 * se)

        std_ok = True
        if std_pt > args.std_ratio_min_std:
            lo, hi = args.std_ratio_bounds
            std_ok = lo <= std_ratio <= hi

        ks_p = float("nan")
        ks_applied = not near_constant
        ks_reject = False
        if ks_applied:
            ks_tested += 1
            _, ks_p = ks_test(pt_k, cpp_k)
            ks_reject = ks_p < bonferroni_alpha
            if ks_reject:
                ks_rejects.append(action_names[k])

        status = "PASS" if (mean_ok and std_ok) else "FAIL"
        if status == "FAIL":
            fail_dims.append(action_names[k])

        ks_flag = ("R" if ks_reject else ".") if ks_applied else "skip"
        ks_p_s = f"{ks_p:9.6f}" if ks_applied else f"{'--':>9}"
        print(f"{action_names[k]:>5} {mean_pt:11.6f} {mean_cpp:11.6f} {dmean:9.6f} {se:9.6f} "
              f"{std_pt:9.6f} {std_cpp:9.6f} {std_ratio:7.3f} {ks_p_s} {ks_flag:>4} {status:>7}")

    print()
    ks_ok = len(ks_rejects) <= args.max_ks_rejects
    mean_std_ok = len(fail_dims) == 0
    verdict = "PASS" if (mean_std_ok and ks_ok and noise_ok) else "FAIL"

    print(f"mean/std per-element: {'PASS' if mean_std_ok else f'FAIL ({len(fail_dims)}/{n_dims}: {fail_dims})'}")
    print(f"KS (tested {ks_tested}/{n_dims}, skipped near-constant): "
          f"{len(ks_rejects)} reject(s) {ks_rejects} -> "
          f"{'PASS' if ks_ok else 'FAIL'} (budget <= {args.max_ks_rejects})")
    print(f"noise sanity: {'PASS' if noise_ok else 'FAIL'}")
    print(f"VERDICT: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
