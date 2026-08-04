# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TIP-07: teacher-forced open-loop evaluation of the Octo L1/proprio (head_type=l1)
checkpoint through vla-server, reusing VlaCppClient (not a new client).

Protocol mirrors octo-pytorch-kamusarj's scripts/evaluate_octo_open_loop.py (TIP-02
Task-4; formulas cross-checked against that script's _masked_metrics/_plot_trajectory/
_aggregate, ~/work/octo-pytorch-kamusarj at commit 47f1a3e): read a real RLDS episode,
run inference every `execution_horizon` dataset steps with the recorded (teacher-forced)
observation, stitch the executed prefix of each predicted chunk into a full-length
prediction, and report normalized + original-unit MAE/MSE/RMSE, gripper accuracy, and a
GT-vs-pred plot.

VlaCppClient's octo path (_predict_chunk_octo) predates this checkpoint's proprio input
(its ARCH_PRESETS["octo"] entry hardcodes max_state_dim=0 for the older, image-only
cyrusneary/octo-finetuned-libero checkpoint) -- it builds images + language but never
reads/sends Inputs::state. Per TIP-07 ("KHONG sua VlaCppClient cu, chi import"),
vla_cpp_client.py itself is untouched; OctoL1Client below subclasses it and overrides
just that one method to also send raw proprio via req.state (mirrors the parent's
image/tokenize/send logic, since Python has no clean way to inject one extra line into a
method without overriding the whole body).

Usage:
    python run_open_loop_octo.py \
        --vla-addr tcp://localhost:5555 \
        --rlds-dir ~/aloha_carrot_ep0_rlds/aloha_carrot_easy_rlds/1.0.0 \
        --dataset-statistics ~/octo_ckpts/kamusarj_jitter2525/dataset_statistics.json \
        --golden-t0-dir ~/octo_l1_golden/ep0_t0 \
        --output-dir outputs/open_loop_octo
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from client.vla_cpp_client import VlaCppClient

ACTION_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
PROTOCOL_SOURCE = (
    "https://github.com/NVIDIA/Isaac-GR00T/blob/main/"
    "getting_started/finetune_new_embodiment.md#step-4-open-loop-evaluation"
)


class OctoL1Client(VlaCppClient):
    """TIP-07: adds raw-proprio state to the octo predict request. See module docstring
    for why this is an override (not an edit) of the parent's _predict_chunk_octo."""

    def _predict_chunk_octo(self, observations: dict[str, Any]) -> np.ndarray:
        images_u8: list[np.ndarray] = []
        for key in self.image_keys[:2]:
            if key not in observations:
                continue
            img = np.asarray(observations[key], dtype=np.uint8)
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"octo: {key} expected HWC uint8 [H,W,3], got {img.shape}")
            images_u8.append(np.ascontiguousarray(img, dtype=np.uint8))
        if not images_u8:
            raise KeyError(f"octo: no image keys found in observations; got {list(observations.keys())}")

        task = observations.get("task", "")
        if isinstance(task, bytes):
            task = task.decode()
        toks = self.tok(task, return_tensors="np", padding="max_length",
                        truncation=True, max_length=self.max_length)
        input_ids = toks["input_ids"][0].astype(np.int32)
        attn_mask = toks["attention_mask"][0].astype(np.int32)

        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1
        for img in images_u8:
            ip = req.images.add()
            ip.encoding = self.pb.Image.RGB_U8
            ip.height = img.shape[0]
            ip.width = img.shape[1]
            ip.data = img.tobytes()
        req.lang_tokens.extend(input_ids.tolist())
        req.attention_mask.extend(attn_mask.tolist())

        # TIP-07 addition: raw (un-normalized) proprio -- octo.cpp's proprio tokenizer
        # z-scores it server-side (TIP-05 convention), matching Inputs::state's doc
        # comment in model.h. self.max_state_dim must be overridden to 7 at construction
        # (the "octo" ARCH_PRESETS default of 0 is for the older proprio-less checkpoint).
        state = observations.get("state")
        if state is not None:
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            if state.shape[0] != self.max_state_dim:
                raise ValueError(
                    f"octo: state has {state.shape[0]} dims, expected max_state_dim={self.max_state_dim}")
            req.state.extend(float(x) for x in state)

        self.sock.send(req.SerializeToString())
        body = self.sock.recv()
        resp = self.pb.PredictResponse()
        resp.ParseFromString(body)
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")
        self._last_response = resp
        return (np.array(resp.action_chunk, dtype=np.float32)
                  .reshape(resp.chunk_size, resp.action_dim))


def load_episode(rlds_dir: Path, traj_index: int) -> list[dict[str, Any]]:
    builder = tfds.builder_from_directory(str(rlds_dir))
    ds = builder.as_dataset(split="train")
    episode = None
    for i, ep in enumerate(ds):
        if i == traj_index:
            episode = ep
            break
    if episode is None:
        raise IndexError(f"{rlds_dir}: no trajectory index {traj_index}")
    steps = []
    for s in episode["steps"]:
        steps.append({
            "top": s["observation"]["top"].numpy(),
            "wrist": s["observation"]["wrist"].numpy(),
            "state": s["observation"]["state"].numpy().astype(np.float32),
            "action": s["action"].numpy().astype(np.float64),
            "instruction": s["language_instruction"].numpy().decode("utf-8"),
        })
    return steps


def zscore_normalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    mask = np.asarray(stats.get("mask", np.ones_like(mean, dtype=bool)), dtype=bool)
    values = np.asarray(values, dtype=np.float64)
    return np.where(mask, (values - mean) / std, values)


def masked_metrics(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    # TIP-02 formula, verbatim from evaluate_octo_open_loop.py::_masked_metrics.
    if predicted.shape != target.shape or predicted.shape != valid.shape:
        raise ValueError(f"metric shapes differ: predicted={predicted.shape} target={target.shape} valid={valid.shape}")
    count = int(valid.sum())
    if count == 0:
        raise ValueError("trajectory contains no valid action targets")
    difference = predicted - target
    abs_error = np.abs(difference)
    sq_error = np.square(difference)
    dim_count = valid.sum(axis=0)
    per_dim_mae = np.divide((abs_error * valid).sum(axis=0), dim_count,
                            out=np.full(predicted.shape[1], np.nan, dtype=np.float64), where=dim_count > 0)
    per_dim_mse = np.divide((sq_error * valid).sum(axis=0), dim_count,
                            out=np.full(predicted.shape[1], np.nan, dtype=np.float64), where=dim_count > 0)
    return {
        "valid_action_values": count,
        "absolute_error_sum": float((abs_error * valid).sum()),
        "squared_error_sum": float((sq_error * valid).sum()),
        "mae": float((abs_error * valid).sum() / count),
        "mse": float((sq_error * valid).sum() / count),
        "rmse": float(math.sqrt((sq_error * valid).sum() / count)),
        "per_dim_mae": per_dim_mae.tolist(),
        "per_dim_mse": per_dim_mse.tolist(),
    }


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    # TIP-02 formula, verbatim from evaluate_octo_open_loop.py::_aggregate.
    result: dict[str, Any] = {"trajectories": len(reports)}
    for key in ("normalized", "original_units"):
        count = sum(r[key]["valid_action_values"] for r in reports)
        absolute_error_sum = sum(r[key]["absolute_error_sum"] for r in reports)
        squared_error_sum = sum(r[key]["squared_error_sum"] for r in reports)
        result[key] = {
            "valid_action_values": count,
            "mae": absolute_error_sum / count,
            "mse": squared_error_sum / count,
            "rmse": math.sqrt(squared_error_sum / count),
        }
    gripper_values = [r["gripper_accuracy"] for r in reports if r["gripper_accuracy"] is not None]
    result["mean_gripper_accuracy"] = float(np.mean(gripper_values)) if gripper_values else None
    return result


def plot_trajectory(*, predicted, target, state, valid, inference_points, title, output_path: Path) -> None:
    # TIP-02 formula, verbatim from evaluate_octo_open_loop.py::_plot_trajectory.
    action_dim = target.shape[1]
    names = ACTION_NAMES[:action_dim] + [f"action_{i}" for i in range(len(ACTION_NAMES), action_dim)]
    fig, axes = plt.subplots(action_dim, 1, figsize=(12, max(4, 2.6 * action_dim)), sharex=True, dpi=140)
    if action_dim == 1:
        axes = [axes]
    timesteps = np.arange(len(target))
    for dim, ax in enumerate(axes):
        target_values = np.where(valid[:, dim], target[:, dim], np.nan)
        predicted_values = np.where(valid[:, dim], predicted[:, dim], np.nan)
        if state is not None and state.shape == target.shape:
            ax.plot(timesteps, state[:, dim], color="#666666", linestyle="--", linewidth=1.3,
                    alpha=0.8, label="state", zorder=1)
        ax.plot(timesteps, target_values, color="#0072B2", linewidth=2.1, label="GT action", zorder=3)
        ax.plot(timesteps, predicted_values, color="#D55E00", linewidth=1.9, label="pred action", zorder=2)
        for point in inference_points:
            ax.axvline(point, color="#CC79A7", linestyle=":", alpha=0.35, linewidth=1.0, zorder=0)
        visible = [p for p in inference_points if 0 <= p < len(predicted_values) and np.isfinite(predicted_values[p])]
        if visible:
            ax.scatter(visible, predicted_values[visible], color="#CC79A7", edgecolors="white",
                      linewidths=0.5, s=24, label="inference point", zorder=4)
        ax.set_ylabel(names[dim])
        ax.grid(True, color="#D0D0D0", alpha=0.45, linewidth=0.7)
        if dim == 0:
            ax.legend(loc="upper right", ncol=4, framealpha=0.95)
    axes[-1].set_xlabel("dataset timestep")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def cross_check_preprocess(steps: list[dict[str, Any]], golden_dir: Path) -> bool:
    """TIP-07 AC: model-input at t=0 (raw images + raw proprio the runner is about to
    send) must match TIP-06's golden dump exactly -- same RLDS episode/step, proving the
    runner reads the identical source data octo_l1_parity's golden was built from. Does
    NOT re-check the server's internal normalize/tokenize forward pass -- that's already
    verified bit-exact (T0-T5, <1e-4) by TIP-06's ctest, independent of this client."""
    ok = True
    golden_top = np.load(golden_dir / "input.top_hwc_u8.npy")
    golden_wrist = np.load(golden_dir / "input.wrist_hwc_u8.npy")
    golden_proprio = np.load(golden_dir / "input.proprio_raw.npy")
    top_diff = int(np.abs(steps[0]["top"].astype(np.int32) - golden_top.astype(np.int32)).max())
    wrist_diff = int(np.abs(steps[0]["wrist"].astype(np.int32) - golden_wrist.astype(np.int32)).max())
    proprio_diff = float(np.abs(steps[0]["state"] - golden_proprio).max())
    print(f"preprocess cross-check vs {golden_dir}:")
    print(f"  top image max|diff|    = {top_diff} (expect 0)")
    print(f"  wrist image max|diff|  = {wrist_diff} (expect 0)")
    print(f"  proprio_raw max|diff|  = {proprio_diff:.8f} (expect < 1e-4)")
    if top_diff != 0 or wrist_diff != 0 or proprio_diff >= 1e-4:
        ok = False
    print(f"  PREPROCESS CROSS-CHECK: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vla-addr", default="tcp://localhost:5555")
    ap.add_argument("--rlds-dir", required=True, type=Path,
                    help="RLDS builder dir, e.g. ~/aloha_carrot_ep0_rlds/aloha_carrot_easy_rlds/1.0.0")
    ap.add_argument("--dataset-statistics", required=True, type=Path,
                    help="jitter2525's dataset_statistics.json (action mean/std/mask/min/max)")
    ap.add_argument("--golden-t0-dir", type=Path, default=None,
                    help="TIP-06 octo_l1_golden/ep0_t0 dir for the preprocess cross-check; skipped if omitted")
    ap.add_argument("--traj-index", type=int, default=0)
    ap.add_argument("--execution-horizon", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None, help="default: full episode length")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--recv-timeout-ms", type=int, default=120_000)
    args = ap.parse_args()

    dataset_statistics = json.loads(args.dataset_statistics.expanduser().read_text())
    action_stats = dataset_statistics["action"]
    action_dim = len(action_stats["mean"])

    print(f"loading RLDS episode {args.traj_index} from {args.rlds_dir} ...")
    rlds_steps = load_episode(args.rlds_dir.expanduser(), args.traj_index)
    trajectory_length = len(rlds_steps)
    print(f"episode has {trajectory_length} steps")

    preprocess_ok = None
    if args.golden_t0_dir is not None:
        preprocess_ok = cross_check_preprocess(rlds_steps, args.golden_t0_dir.expanduser())

    client = OctoL1Client(
        vla_addr=args.vla_addr,
        arch="octo",
        max_state_dim=action_dim,  # override ARCH_PRESETS["octo"]'s stale max_state_dim=0
        real_action_dim=action_dim,
        image_keys=["observation.images.image", "observation.images.image2"],
        recv_timeout_ms=args.recv_timeout_ms,
    )

    actual_steps = min(args.steps, trajectory_length) if args.steps else trajectory_length
    predicted = np.full((actual_steps, action_dim), np.nan, dtype=np.float64)
    target = np.array([s["action"] for s in rlds_steps[:actual_steps]], dtype=np.float64)
    state_units = np.array([s["state"] for s in rlds_steps[:actual_steps]], dtype=np.float64)
    valid = np.zeros((actual_steps, action_dim), dtype=np.bool_)
    inference_points: list[int] = []
    instruction = rlds_steps[0]["instruction"]

    for t in range(0, actual_steps, args.execution_horizon):
        step = rlds_steps[t]
        observations = {
            "observation.images.image": step["top"],
            "observation.images.image2": step["wrist"],
            "task": instruction,
            "state": step["state"],
        }
        chunk = client._predict_chunk_octo(observations)  # (20,7), already unnormalized
        take = min(args.execution_horizon, actual_steps - t, chunk.shape[0])
        predicted[t:t + take] = chunk[:take, :action_dim]
        valid[t:t + take] = True
        inference_points.append(t)
        print(f"t={t:3d} take={take} chunk[0]={chunk[0, :action_dim].round(4).tolist()}")

    n_calls = len(inference_points)
    print(f"inference_calls={n_calls} evaluated_steps={actual_steps}")

    predicted_norm = zscore_normalize(predicted, action_stats)
    target_norm = zscore_normalize(target, action_stats)
    normalized_metrics = masked_metrics(predicted_norm, target_norm, valid)
    original_metrics = masked_metrics(predicted, target, valid)

    stats_min = np.asarray(action_stats["min"], dtype=np.float64)
    stats_max = np.asarray(action_stats["max"], dtype=np.float64)
    close_threshold = float((stats_min[-1] + stats_max[-1]) / 2.0)
    gripper_valid = valid[:, -1]
    predicted_closed = predicted[:, -1] <= close_threshold
    target_closed = target[:, -1] <= close_threshold
    gripper_accuracy = float(np.mean(predicted_closed[gripper_valid] == target_closed[gripper_valid])) \
        if gripper_valid.any() else None

    output_dir = args.output_dir.expanduser()
    split_output = output_dir / "train"
    split_output.mkdir(parents=True, exist_ok=True)
    stem = "trajectory_000"
    plot_path = split_output / f"{stem}_gt_vs_pred.png"
    trace_path = split_output / f"{stem}_actions.npz"

    plot_trajectory(
        predicted=predicted, target=target, state=state_units, valid=valid,
        inference_points=inference_points,
        title=f"octo-aloha-jitter2525 | train trajectory {args.traj_index} | execution horizon {args.execution_horizon}",
        output_path=plot_path,
    )
    np.savez_compressed(
        trace_path,
        predicted_actions=predicted,
        ground_truth_actions=target,
        predicted_actions_normalized=predicted_norm,
        ground_truth_actions_normalized=target_norm,
        valid_action_mask=valid,
        inference_points=np.asarray(inference_points, dtype=np.int32),
        state_actions_units=state_units,
    )

    report = {
        "split": "train",
        "trajectory_index": args.traj_index,
        "source_episode_id": args.traj_index,
        "trajectory_length": trajectory_length,
        "evaluated_steps": actual_steps,
        "inference_calls": n_calls,
        "execution_horizon": args.execution_horizon,
        "instruction": instruction,
        "normalized": normalized_metrics,
        "original_units": original_metrics,
        "gripper_accuracy": gripper_accuracy,
        "plot": str(plot_path.resolve()),
        "action_trace": str(trace_path.resolve()),
    }
    reports = [report]
    agg = aggregate(reports)
    summary = {
        "protocol": "teacher_forced_open_loop_action_chunk_stitching",
        "protocol_source": PROTOCOL_SOURCE,
        "vla_addr": args.vla_addr,
        "rlds_dir": str(args.rlds_dir.resolve()),
        "dataset_statistics": str(args.dataset_statistics.resolve()),
        "traj_index": args.traj_index,
        "action_horizon": 20,
        "execution_horizon": args.execution_horizon,
        "requested_steps": args.steps,
        "preprocess_cross_check_pass": preprocess_ok,
        "aggregate": agg,
        "split_aggregates": {"train": agg},
        "trajectories": reports,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    csv_path = output_dir / "trajectory_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        fieldnames = ["split", "trajectory_index", "source_episode_id", "trajectory_length",
                      "evaluated_steps", "inference_calls", "execution_horizon",
                      "normalized_mae", "normalized_mse", "original_mae", "original_mse",
                      "gripper_accuracy", "plot", "action_trace"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "split": report["split"], "trajectory_index": report["trajectory_index"],
            "source_episode_id": report["source_episode_id"], "trajectory_length": report["trajectory_length"],
            "evaluated_steps": report["evaluated_steps"], "inference_calls": report["inference_calls"],
            "execution_horizon": report["execution_horizon"],
            "normalized_mae": report["normalized"]["mae"], "normalized_mse": report["normalized"]["mse"],
            "original_mae": report["original_units"]["mae"], "original_mse": report["original_units"]["mse"],
            "gripper_accuracy": report["gripper_accuracy"],
            "plot": report["plot"], "action_trace": report["action_trace"],
        })

    print(f"summary={output_dir / 'summary.json'}")
    print(f"metrics_csv={csv_path}")
    print(f"original: mae={original_metrics['mae']:.6f} mse={original_metrics['mse']:.6f} rmse={original_metrics['rmse']:.6f}")
    print(f"normalized: mae={normalized_metrics['mae']:.6f} mse={normalized_metrics['mse']:.6f} rmse={normalized_metrics['rmse']:.6f}")
    print(f"per_dim_mae={[round(v, 4) for v in original_metrics['per_dim_mae']]}")
    print(f"gripper_accuracy={gripper_accuracy}")


if __name__ == "__main__":
    main()
