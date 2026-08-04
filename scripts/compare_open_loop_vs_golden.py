#!/usr/bin/env python3
"""TIP-08: direct comparison of vla.cpp's open-loop output (TIP-07) against the local
OctoPt golden trace (TIP-02) for the SAME RLDS trajectory -- the most direct parity check
available (same model weights, same input, only the inference engine differs).

Compares npz arrays directly (predicted_actions, predicted_actions_normalized,
ground_truth_actions, valid_action_mask), recomputes MAE/MSE/RMSE/gripper/per-dim/
normalized metrics for both sides with the identical formula (mirrors
evaluate_octo_open_loop.py's _masked_metrics, already duplicated once in
run_open_loop_octo.py for TIP-07), and renders a 7-subplot overlay plot.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

ACTION_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]


def masked_metrics(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    # Verbatim formula, TIP-02's evaluate_octo_open_loop.py::_masked_metrics.
    count = int(valid.sum())
    difference = predicted - target
    abs_error = np.abs(difference)
    sq_error = np.square(difference)
    dim_count = valid.sum(axis=0)
    per_dim_mae = np.divide((abs_error * valid).sum(axis=0), dim_count,
                            out=np.full(predicted.shape[1], np.nan), where=dim_count > 0)
    per_dim_mse = np.divide((sq_error * valid).sum(axis=0), dim_count,
                            out=np.full(predicted.shape[1], np.nan), where=dim_count > 0)
    return {
        "valid_action_values": count,
        "mae": float((abs_error * valid).sum() / count),
        "mse": float((sq_error * valid).sum() / count),
        "rmse": float(math.sqrt((sq_error * valid).sum() / count)),
        "per_dim_mae": per_dim_mae.tolist(),
        "per_dim_mse": per_dim_mse.tolist(),
    }


def gripper_accuracy(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray, threshold: float) -> float:
    gripper_valid = valid[:, -1]
    predicted_closed = predicted[:, -1] <= threshold
    target_closed = target[:, -1] <= threshold
    return float(np.mean(predicted_closed[gripper_valid] == target_closed[gripper_valid]))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla-npz", type=Path, required=True)
    ap.add_argument("--golden-npz", type=Path, required=True)
    ap.add_argument("--golden-metrics-json", type=Path, default=None,
                    help="golden's own trajectory_000_metrics.json (published target)")
    ap.add_argument("--dataset-statistics", type=Path, required=True)
    ap.add_argument("--output-plot", type=Path, required=True)
    args = ap.parse_args()

    vla = np.load(args.vla_npz.expanduser())
    golden = np.load(args.golden_npz.expanduser())

    action_stats = json.loads(args.dataset_statistics.expanduser().read_text())["action"]
    stats_min = np.asarray(action_stats["min"], dtype=np.float64)
    stats_max = np.asarray(action_stats["max"], dtype=np.float64)
    gripper_threshold = float((stats_min[-1] + stats_max[-1]) / 2.0)

    print("=== 1. Direct array comparison (vla.cpp vs local golden) ===")
    print(f"{'array':<32}{'max|diff|':>14}")
    array_diffs = {}
    for key in ("predicted_actions", "predicted_actions_normalized", "ground_truth_actions", "valid_action_mask"):
        diff = max_abs_diff(vla[key], golden[key])
        array_diffs[key] = diff
        print(f"{key:<32}{diff:>14.8f}")

    valid = golden["valid_action_mask"].astype(bool)
    vla_metrics = masked_metrics(vla["predicted_actions"], vla["ground_truth_actions"], valid)
    vla_metrics_norm = masked_metrics(vla["predicted_actions_normalized"], vla["ground_truth_actions_normalized"], valid)
    vla_gripper = gripper_accuracy(vla["predicted_actions"], vla["ground_truth_actions"], valid, gripper_threshold)

    golden_metrics = masked_metrics(golden["predicted_actions"], golden["ground_truth_actions"], valid)
    golden_metrics_norm = masked_metrics(golden["predicted_actions_normalized"], golden["ground_truth_actions_normalized"], valid)
    golden_gripper = gripper_accuracy(golden["predicted_actions"], golden["ground_truth_actions"], valid, gripper_threshold)

    published = None
    if args.golden_metrics_json is not None:
        published = json.loads(args.golden_metrics_json.expanduser().read_text())

    print()
    print("=== 2. Metric comparison: vla.cpp vs local golden (recomputed) vs published target ===")
    header = f"{'metric':<20}{'vla.cpp':>14}{'golden(npz)':>14}{'published':>14}{'|vla-golden|':>15}{'|vla-pub|':>13}"
    print(header)
    rows = [
        ("original.mae", vla_metrics["mae"], golden_metrics["mae"],
         published["original_units"]["mae"] if published else None),
        ("original.mse", vla_metrics["mse"], golden_metrics["mse"],
         published["original_units"]["mse"] if published else None),
        ("original.rmse", vla_metrics["rmse"], golden_metrics["rmse"],
         published["original_units"]["rmse"] if published else None),
        ("normalized.mae", vla_metrics_norm["mae"], golden_metrics_norm["mae"],
         published["normalized"]["mae"] if published else None),
        ("normalized.mse", vla_metrics_norm["mse"], golden_metrics_norm["mse"],
         published["normalized"]["mse"] if published else None),
        ("normalized.rmse", vla_metrics_norm["rmse"], golden_metrics_norm["rmse"],
         published["normalized"]["rmse"] if published else None),
        ("gripper_accuracy", vla_gripper, golden_gripper,
         published["gripper_accuracy"] if published else None),
    ]
    for name, v, g, p in rows:
        d_vg = abs(v - g)
        d_vp = abs(v - p) if p is not None else float("nan")
        p_str = f"{p:14.6f}" if p is not None else f"{'n/a':>14}"
        print(f"{name:<20}{v:14.6f}{g:14.6f}{p_str}{d_vg:15.8f}{d_vp:13.8f}")

    print()
    print("per_dim_mae (original units):")
    print(f"{'dim':<10}{'vla.cpp':>12}{'golden':>12}{'|diff|':>12}")
    for i, name in enumerate(ACTION_NAMES):
        v = vla_metrics["per_dim_mae"][i]
        g = golden_metrics["per_dim_mae"][i]
        print(f"{name:<10}{v:12.6f}{g:12.6f}{abs(v - g):12.8f}")

    # 3. Overlay plot: GT, vla.cpp pred, golden pred per dim.
    action_dim = vla["ground_truth_actions"].shape[1]
    fig, axes = plt.subplots(action_dim, 1, figsize=(12, max(4, 2.6 * action_dim)), sharex=True, dpi=140)
    timesteps = np.arange(vla["ground_truth_actions"].shape[0])
    for dim, ax in enumerate(axes):
        gt = np.where(valid[:, dim], vla["ground_truth_actions"][:, dim], np.nan)
        pred_vla = np.where(valid[:, dim], vla["predicted_actions"][:, dim], np.nan)
        pred_golden = np.where(valid[:, dim], golden["predicted_actions"][:, dim], np.nan)
        ax.plot(timesteps, gt, color="#0072B2", linewidth=2.1, label="GT action", zorder=3)
        ax.plot(timesteps, pred_golden, color="#009E73", linewidth=1.6, label="pred (OctoPt golden)",
                zorder=2, alpha=0.9)
        ax.plot(timesteps, pred_vla, color="#D55E00", linewidth=1.4, linestyle="--",
                label="pred (vla.cpp)", zorder=4, alpha=0.9)
        ax.set_ylabel(ACTION_NAMES[dim])
        ax.grid(True, color="#D0D0D0", alpha=0.45, linewidth=0.7)
        if dim == 0:
            ax.legend(loc="upper right", ncol=3, framealpha=0.95)
    axes[-1].set_xlabel("dataset timestep")
    fig.suptitle("vla.cpp vs OctoPt golden -- open-loop prediction overlay (TIP-08)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot)
    plt.close(fig)
    print()
    print(f"overlay plot: {args.output_plot}")

    # Gate check.
    ok = True
    if array_diffs["predicted_actions"] >= 1e-4:
        ok = False
    if array_diffs["ground_truth_actions"] != 0.0 or array_diffs["valid_action_mask"] != 0.0:
        ok = False
    print()
    print(f"G8 array-diff gate: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
