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

"""TIP-11B: UR10e open-loop evaluation of vla.cpp's GR00T-N1.7 GGUF checkpoint, reusing
Isaac-GR00T's OWN gr00t.eval.open_loop_eval.evaluate_single_trajectory (yennt @ 3695c46)
instead of re-deriving observation construction / chunk stitching / MSE-MAE / plotting.

Two thin pieces are added here:

1. Gr00tUR10eClient(VlaCppClient) -- overrides only _predict_chunk_gr00t_n1_7 to use
   UR10e's image/state keys instead of LIBERO's. See the class docstring below for the
   exact diff against the parent method (vla_cpp_client.py, commit
   0d8742049027d4f83be92772686ed7a4f206ece6, lines 1139-1228). vla_cpp_client.py itself is
   NOT modified, per project convention (run_open_loop_octo.py's "KHONG sua VlaCppClient
   cu, chi import").

2. VlaCppPolicy(BasePolicy) -- wraps the client so evaluate_single_trajectory's
   `policy.get_action(parsed_obs)` contract is satisfied. Its unnormalize_chunk() (module
   level) implements the relative->absolute formula extracted from Isaac-GR00T source in
   TIP-11B Section 1 (see the Completion Report for the full citation trail); the short
   version:
     - single_arm is ActionRepresentation.RELATIVE: the model's raw output is a per-step
       (16 steps x 6 dims) normalized delta. state_action_processor.py's relative_action
       override (line 201) copies the ENTIRE relative_action stats blob (min/max/mean/std/
       q01/q99) into norm_params verbatim, bypassing the use_percentiles branch entirely --
       so unnormalize_values_minmax (gr00t/data/utils.py:151) reads the literal "min"/"max"
       fields of relative_action.single_arm, NOT q01/q99, even though use_percentiles=True
       globally (processor/processor_config.json). The per-step delta is then added to the
       CURRENT state (JointActionChunk.to_absolute_chunking, action_chunking.py:384) -- one
       reference state broadcast across all 16 steps, not a running per-step integration.
     - gripper is ActionRepresentation.ABSOLUTE: it never enters the relative_action
       override, so it takes the normal use_percentiles=True path -- literal q01/q99 of
       dataset_statistics.json["new_embodiment"]["action"]["gripper"], no state addition.

evaluate_single_trajectory computes gt_action_across_time / pred_action_across_time /
state_joints_across_time internally but only RETURNS (mse, mae) -- it hands those three
arrays to gr00t.eval.open_loop_eval.plot_trajectory_results (unmodified) to draw the plot.
To save a golden-schema-compatible traj_<id>.npz (gt, pred, state, mse, mae, action_keys)
without reimplementing that internal loop, main() temporarily monkeypatches
gr00t.eval.open_loop_eval.plot_trajectory_results with a thin wrapper that captures its
kwargs as a side channel and then calls the REAL (unmodified) plot_trajectory_results to
still produce the .jpeg. No cut/concat/MSE/MAE/plot math is reimplemented anywhere in this
file.

TIP-11E adds --noise-npy: an optional fixed initial-noise array forwarded to the server via
PredictRequest.noise (vla.proto:26, already wired end-to-end server.cpp:336-435 ->
gr00tn1d7.cpp:787 -- no C++/protobuf changes needed). Lets the C++ engine and a
correspondingly-monkeypatched Isaac-GR00T golden run start the 4-step Euler denoise from the
identical sample, isolating cross-engine numerical drift from independent-noise-sampling
variance. Default None: unchanged TIP-11B behavior (server samples its own N(0,1)).

Usage:
    python run_open_loop_gr00t.py \
        --vla-addr tcp://localhost:5555 \
        --dataset-path /mnt/d/vla_ur10e/data/ur10e-cup \
        --stats-json /mnt/d/vla_ur10e/ckpt/ur10e_cup_v10/experiment_cfg/dataset_statistics.json \
        --traj-ids 65 66 67 --split val --execution-horizon 16 --steps 10000 \
        --output-dir /mnt/d/vla_ur10e/outputs/vlacpp
"""

from __future__ import annotations

import argparse
import csv
import json
import runpy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from client.vla_cpp_client import VlaCppClient

import gr00t.eval.open_loop_eval as open_loop_eval_module
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import evaluate_single_trajectory
from gr00t.policy import BasePolicy

DEFAULT_UR10E_CONFIG = "~/work/Isaac-GR00T/examples/UR10e/ur10e_config.py"


class Gr00tUR10eClient(VlaCppClient):
    """Derived from vla_cpp_client.py::VlaCppClient._predict_chunk_gr00t_n1_7
    @ commit 0d8742049027d4f83be92772686ed7a4f206ece6, lines 1139-1228.

    Changed vs. that LIBERO method:
      - image keys : ("video.image", "video.wrist_image") -> ("video.side", "video.wrist")
      - state keys : x/y/z/roll/pitch/yaw/gripper (dims 1,1,1,1,1,1,2)
                     -> single_arm/gripper (dims 6,1)
      - state stats: LIBERO blob[key]["state"] -> dataset_statistics.json
                     ["new_embodiment"]["state"], order single_arm -> gripper. Read here
                     directly (self._ur10e_state_q01/q99) instead of via the parent's
                     self._gr00t_state_norm hook, because __init__ is called below with
                     stats_json=None -- passing UR10e's stats_json through would hit the
                     parent's LIBERO-schema-specific block (vla_cpp_client.py:294-353,
                     which expects blob[key]["action"]["x"/"y"/"z"/...] and would KeyError
                     on UR10e's {state, action, relative_action} schema).
      - action unnorm: NOT done here (self._gr00t_action_unnorm also stays None for the
                     same reason) -- this method returns the RAW (40, 132) server chunk
                     unmodified. unnormalize_chunk() (module-level, below) does the UR10e
                     relative-single_arm + absolute-gripper unnormalization.

    Reused verbatim from the parent: self._gr00t_eval_image_transform (256x256, crop 0.95,
    static method, untouched), and the entire protobuf request-building / tokenize /
    send-recv block.

    NOT copied from Gr00tPipelineAdapter (LIBERO/robosuite-specific, wrong for UR10e -- see
    TIP-11B Section 0.4):
      - 180-degree image flip ([::-1, ::-1])
      - gripper sign flip (action[6] = -1 if >0.5 else 1)
    """

    _UR10E_STATE_KEYS = ("single_arm", "gripper")
    _UR10E_STATE_DIMS = (6, 1)

    def __init__(self, vla_addr: str = "tcp://localhost:5555", *,
                 state_q01: np.ndarray, state_q99: np.ndarray,
                 noise_flat: np.ndarray | None = None, **kwargs):
        kwargs.pop("stats_json", None)
        kwargs.pop("arch", None)
        super().__init__(vla_addr, arch="gr00t_n1_7", stats_json=None, **kwargs)
        self._ur10e_state_q01 = np.asarray(state_q01, dtype=np.float32).reshape(-1)
        self._ur10e_state_q99 = np.asarray(state_q99, dtype=np.float32).reshape(-1)
        rng = (self._ur10e_state_q99 - self._ur10e_state_q01).astype(np.float32)
        self._ur10e_state_rng = np.where(rng > 1e-8, rng, 1.0).astype(np.float32)
        print(f"vla-cpp-direct[arch=gr00t_n1_7/ur10e]: state normalizer (q01/q99 + clip) "
              f"q01={self._ur10e_state_q01.tolist()} q99={self._ur10e_state_q99.tolist()}",
              flush=True)

        # TIP-11E: caller-supplied initial noise for the action expert, injected into
        # req.noise so C++ takes gr00tn1d7.cpp:787's `if (in.noise) memcpy(...)` path
        # instead of sampling its own N(0,1). None (default) preserves TIP-11B behavior
        # unchanged (server samples its own noise). Layout per TIP-11E Section B0: the
        # server uploads this flat buffer byte-for-byte into a ggml tensor created as
        # ggml_new_tensor_2d(C, GGML_TYPE_F32, AD, AH) (gr00tn1d7.cpp:806) -- ne[0]=AD=132
        # is the fastest-varying dim -- so the flat array must be numpy (40, 132) C-order
        # (AD fastest), no transpose.
        self._noise_flat: np.ndarray | None = None
        if noise_flat is not None:
            self._noise_flat = np.ascontiguousarray(noise_flat, dtype=np.float32).reshape(-1)
            if self._noise_flat.size != 5280:
                raise ValueError(
                    f"noise_flat must have 5280 elements (action_horizon=40 * "
                    f"max_action_dim=132), got {self._noise_flat.size}")
            print(f"[noise] n={self._noise_flat.size} sum={float(self._noise_flat.sum()):.6f}",
                  flush=True)

    def _predict_chunk_gr00t_n1_7(self, observations: dict[str, Any]) -> np.ndarray:
        import re

        images_f32: list[np.ndarray] = []
        for key in ("video.side", "video.wrist"):
            if key not in observations:
                raise KeyError(
                    f"gr00t_n1_7/ur10e image key '{key}' missing; got {list(observations.keys())}")
            img = observations[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            img = np.asarray(img, dtype=np.uint8)
            while img.ndim > 3:
                img = img[0]
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"{key}: expected HWC u8 [H, W, 3], got {img.shape} dtype={img.dtype}")
            img_u8 = self._gr00t_eval_image_transform(
                img, self._GR00T_TARGET_SIZE, self._GR00T_SHORTEST_EDGE, self._GR00T_CROP_FRACTION)

            img_f32 = np.ascontiguousarray(img_u8.astype(np.float32) / 255.0)
            images_f32.append(img_f32)

        state_chunks = []
        for key, dim in zip(self._UR10E_STATE_KEYS, self._UR10E_STATE_DIMS):
            mk = f"state.{key}"
            if mk not in observations:
                raise KeyError(f"gr00t_n1_7/ur10e state key '{mk}' missing; got {list(observations.keys())}")
            v = observations[mk]
            if isinstance(v, torch.Tensor):
                v = v.numpy()
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.size != dim:
                raise ValueError(f"gr00t_n1_7/ur10e state '{mk}': expected {dim}-d, got {v.size}-d")
            state_chunks.append(v)
        state_raw = np.concatenate(state_chunks, axis=0).astype(np.float32)

        state_norm = 2.0 * (state_raw - self._ur10e_state_q01) / self._ur10e_state_rng - 1.0
        state_norm = np.clip(state_norm, -1.0, 1.0).astype(np.float32)
        state_padded = np.zeros(self.max_state_dim, dtype=np.float32)
        state_padded[: state_norm.size] = state_norm

        task_field = observations.get("task", "")
        if isinstance(task_field, tuple):
            task_field = task_field[0] if task_field else ""
        if isinstance(task_field, bytes):
            task_field = task_field.decode()
        task = task_field
        language = re.sub(r"[^\w\s]", "", task.lower())
        n_views = len(images_f32)
        conv = [{"role": "user", "content": [
            *[{"type": "image"} for _ in range(n_views)],
            {"type": "text", "text": language},
        ]}]
        text = self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        expanded: list[int] = []
        for tid in ids:
            if tid == self._GR00T_IMG_PAD_ID:
                expanded.extend([self._GR00T_IMG_PAD_ID] * self._GR00T_N_TOK_PER_VIEW)
            else:
                expanded.append(tid)
        lang = np.array(expanded, dtype=np.int32)

        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1
        for img in images_f32:
            ip = req.images.add()
            ip.encoding = self.pb.Image.F32_RGB_01
            ip.height = img.shape[0]
            ip.width = img.shape[1]
            ip.data = img.tobytes()
        req.lang_tokens.extend(int(t) for t in lang)
        req.state.extend(float(x) for x in state_padded)
        if self._noise_flat is not None:
            req.noise.extend(float(x) for x in self._noise_flat)

        self.sock.send(req.SerializeToString())
        body = self.sock.recv()
        resp = self.pb.PredictResponse()
        resp.ParseFromString(body)
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")
        self._last_response = resp
        chunk = (np.array(resp.action_chunk, dtype=np.float32)
                   .reshape(resp.chunk_size, resp.action_dim))
        return chunk  # RAW, NOT unnormalized -- see unnormalize_chunk() below


def unnormalize_chunk(raw_132: np.ndarray, state_now_7: np.ndarray,
                       norm_params: dict[str, np.ndarray]) -> np.ndarray:
    """TIP-11B Section 1 formula. raw_132: (>=16, 132) raw server output, CHUA
    unnormalize. state_now_7: (7,) current raw state [single_arm(6), gripper(1)].
    norm_params: single_arm_min/max (16,6) literal min/max of relative_action.single_arm
    (NOT q01/q99 -- see class docstring above and the Completion Report Section 1 for the
    state_action_processor.py line citations), gripper_q01/q99 (1,).

    Returns (16, 7) absolute [single_arm(6), gripper(1)]. Only the first 16 rows of
    raw_132 are used: relative_action stats are indexed 0..15 (one bound per horizon
    step), so there is no denorm reference for steps beyond 16 even though the server
    returns 40.
    """
    if raw_132.shape[0] < 16:
        raise ValueError(f"unnormalize_chunk: need >=16 rows, got {raw_132.shape}")
    a16 = np.asarray(raw_132[:16, :7], dtype=np.float64)  # (16,7)

    rel_norm = np.clip(a16[:, :6], -1.0, 1.0)
    sa_min = np.asarray(norm_params["single_arm_min"], dtype=np.float64)  # (16,6)
    sa_max = np.asarray(norm_params["single_arm_max"], dtype=np.float64)  # (16,6)
    rel_abs = (rel_norm + 1.0) / 2.0 * (sa_max - sa_min) + sa_min          # (16,6) delta
    single_arm_now = np.asarray(state_now_7[:6], dtype=np.float64)
    single_arm_abs = single_arm_now[None, :] + rel_abs                    # (16,6)

    grip_norm = np.clip(a16[:, 6:7], -1.0, 1.0)
    g_q01 = np.asarray(norm_params["gripper_q01"], dtype=np.float64)      # (1,)
    g_q99 = np.asarray(norm_params["gripper_q99"], dtype=np.float64)      # (1,)
    gripper_abs = (grip_norm + 1.0) / 2.0 * (g_q99 - g_q01) + g_q01       # (16,1)

    return np.concatenate([single_arm_abs, gripper_abs], axis=1).astype(np.float32)


class VlaCppPolicy(BasePolicy):
    """Thin wrapper satisfying evaluate_single_trajectory's `policy.get_action(parsed_obs)`
    contract (gr00t/eval/open_loop_eval.py:180, gr00t/policy/policy.py:20 BasePolicy).
    strict=False: check_observation/check_action are no-ops here (UR10e's parsed_obs shape
    is fixed by ur10e_config.py + parse_observation_gr00t; validating it again here would
    just duplicate that config)."""

    def __init__(self, client: Gr00tUR10eClient, norm_params: dict[str, np.ndarray],
                 modality_configs: dict[str, Any]):
        super().__init__(strict=False)
        self.client = client
        self._norm_params = norm_params
        self._mc = modality_configs

    def check_observation(self, observation: dict[str, Any]) -> None:
        pass

    def check_action(self, action: dict[str, Any]) -> None:
        pass

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {}

    def _get_action(self, observation: dict[str, Any],
                     options: dict[str, Any] | None = None):
        video = observation["video"]
        state = observation["state"]
        language = observation["language"]

        side = np.asarray(video["side"])[0, -1]
        wrist = np.asarray(video["wrist"])[0, -1]
        single_arm_now = np.asarray(state["single_arm"])[0, -1].astype(np.float32)
        gripper_now = np.asarray(state["gripper"])[0, -1].astype(np.float32)
        state_now_7 = np.concatenate([single_arm_now, gripper_now])

        task_key = self._mc["language"].modality_keys[0]
        task_text = language[task_key][0][0]

        client_obs = {
            "video.side": side,
            "video.wrist": wrist,
            "state.single_arm": single_arm_now,
            "state.gripper": gripper_now,
            "task": task_text,
        }
        raw_132 = self.client._predict_chunk_gr00t_n1_7(client_obs)  # (40, 132) raw
        abs_16x7 = unnormalize_chunk(raw_132, state_now_7, self._norm_params)  # (16, 7)

        action = {
            "single_arm": abs_16x7[None, :, 0:6],  # (1, 16, 6)
            "gripper": abs_16x7[None, :, 6:7],      # (1, 16, 1)
        }
        return action, {}

    def get_modality_config(self) -> dict[str, Any]:
        return self._mc


def load_ur10e_modality_config(config_path: Path) -> dict[str, Any]:
    ns = runpy.run_path(str(config_path))
    if "ur10e_config" not in ns:
        raise KeyError(f"{config_path} has no top-level 'ur10e_config' dict")
    return ns["ur10e_config"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vla-addr", default="tcp://localhost:5555")
    ap.add_argument("--dataset-path", required=True, type=Path,
                    help="LeRobot-format dataset root, e.g. /mnt/d/vla_ur10e/data/ur10e-cup")
    ap.add_argument("--stats-json", required=True, type=Path,
                    help="<ckpt>/experiment_cfg/dataset_statistics.json")
    ap.add_argument("--ur10e-config", default=DEFAULT_UR10E_CONFIG, type=Path,
                    help="Isaac-GR00T examples/UR10e/ur10e_config.py to runpy for modality_configs")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--traj-ids", required=True, type=int, nargs="+")
    ap.add_argument("--split", required=True, choices=["val", "train"])
    ap.add_argument("--execution-horizon", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--recv-timeout-ms", type=int, default=120_000)
    ap.add_argument("--model-label", default=None,
                    help="Plot-title label; default: the --stats-json checkpoint dir name")
    ap.add_argument("--noise-npy", default=None, type=Path,
                    help="TIP-11E: .npy with the action expert's fixed initial noise, "
                         "shape (40, 132) C-order (AD fastest -- see class docstring on "
                         "Gr00tUR10eClient). Default None: server samples its own N(0,1) "
                         "per call, unchanged TIP-11B behavior.")
    args = ap.parse_args()

    stats_path = args.stats_json.expanduser()
    model_label = args.model_label or stats_path.parent.parent.name
    blob = json.loads(stats_path.read_text())
    if args.embodiment_tag not in blob:
        raise KeyError(f"{stats_path} has top-level keys {list(blob)}; expected {args.embodiment_tag!r}")
    stats = blob[args.embodiment_tag]

    norm_params = {
        "single_arm_min": np.asarray(stats["relative_action"]["single_arm"]["min"], dtype=np.float64),
        "single_arm_max": np.asarray(stats["relative_action"]["single_arm"]["max"], dtype=np.float64),
        "gripper_q01": np.asarray(stats["action"]["gripper"]["q01"], dtype=np.float64),
        "gripper_q99": np.asarray(stats["action"]["gripper"]["q99"], dtype=np.float64),
    }
    state_q01 = np.concatenate([
        np.asarray(stats["state"]["single_arm"]["q01"], dtype=np.float32),
        np.asarray(stats["state"]["gripper"]["q01"], dtype=np.float32),
    ])
    state_q99 = np.concatenate([
        np.asarray(stats["state"]["single_arm"]["q99"], dtype=np.float32),
        np.asarray(stats["state"]["gripper"]["q99"], dtype=np.float32),
    ])
    print(f"single_arm relative min[0]={norm_params['single_arm_min'][0].tolist()} "
          f"max[0]={norm_params['single_arm_max'][0].tolist()}", flush=True)
    print(f"gripper action q01={norm_params['gripper_q01'].tolist()} "
          f"q99={norm_params['gripper_q99'].tolist()}", flush=True)

    modality_configs = load_ur10e_modality_config(args.ur10e_config.expanduser())

    noise_flat = None
    if args.noise_npy is not None:
        noise_flat = np.load(args.noise_npy.expanduser()).astype(np.float32)

    client = Gr00tUR10eClient(
        vla_addr=args.vla_addr,
        state_q01=state_q01,
        state_q99=state_q99,
        noise_flat=noise_flat,
        recv_timeout_ms=args.recv_timeout_ms,
    )
    policy = VlaCppPolicy(client, norm_params, modality_configs)

    print(f"loading LeRobotEpisodeLoader from {args.dataset_path} ...", flush=True)
    loader = LeRobotEpisodeLoader(
        dataset_path=str(args.dataset_path.expanduser()),
        modality_configs=modality_configs,
    )
    print(f"dataset length: {len(loader)}", flush=True)

    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)

    output_dir = args.output_dir.expanduser() / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    # Capture (state, gt, pred, action_keys) from evaluate_single_trajectory's internal
    # call to plot_trajectory_results as a side channel -- see module docstring. The real
    # plot_trajectory_results still runs (produces the .jpeg); nothing is reimplemented.
    _original_plot_trajectory_results = open_loop_eval_module.plot_trajectory_results
    _capture: dict[str, Any] = {}

    def _capturing_plot_trajectory_results(**kwargs):
        _capture["state"] = kwargs["state_joints_across_time"]
        _capture["gt"] = kwargs["gt_action_across_time"]
        _capture["pred"] = kwargs["pred_action_across_time"]
        _capture["action_keys"] = kwargs["action_keys"]
        return _original_plot_trajectory_results(**kwargs)

    csv_rows: list[dict[str, Any]] = []
    open_loop_eval_module.plot_trajectory_results = _capturing_plot_trajectory_results
    try:
        for traj_id in args.traj_ids:
            if traj_id >= len(loader):
                print(f"WARNING: traj_id {traj_id} out of range (dataset length {len(loader)}); skipping")
                continue
            plot_path = output_dir / f"traj_{traj_id}.jpeg"
            npz_path = output_dir / f"traj_{traj_id}.npz"

            print(f"=== traj_id={traj_id} split={args.split} ===", flush=True)
            _capture.clear()
            mse, mae = evaluate_single_trajectory(
                policy,
                loader,
                traj_id,
                embodiment_tag,
                None,  # modality_keys: use loader.modality_configs["action"].modality_keys
                steps=args.steps,
                execution_horizon=args.execution_horizon,
                save_plot_path=str(plot_path),
            )
            gt = np.asarray(_capture["gt"])
            pred = np.asarray(_capture["pred"])
            state = np.asarray(_capture["state"])
            action_keys = _capture["action_keys"]
            n_steps = gt.shape[0]

            np.savez_compressed(
                npz_path,
                gt=gt, pred=pred, state=state, mse=np.float64(mse), mae=np.float64(mae),
                action_keys=np.array(action_keys, dtype=object),
            )
            print(f"traj_{traj_id}: n_steps={n_steps} mse={mse:.8f} mae={mae:.8f} "
                  f"pred.mean(axis=0)={pred.mean(axis=0).round(4).tolist()}", flush=True)
            csv_rows.append({
                "split": args.split, "traj_id": traj_id,
                "mse": float(mse), "mae": float(mae), "n_steps": int(n_steps),
            })
    finally:
        open_loop_eval_module.plot_trajectory_results = _original_plot_trajectory_results

    csv_path = output_dir / f"vlacpp_{args.split}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "traj_id", "mse", "mae", "n_steps"])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    if csv_rows:
        avg_mse = float(np.mean([r["mse"] for r in csv_rows]))
        avg_mae = float(np.mean([r["mae"] for r in csv_rows]))
        print(f"[{model_label}] split={args.split} trajectories={len(csv_rows)} "
              f"avg_mse={avg_mse:.8f} avg_mae={avg_mae:.8f}", flush=True)
    print(f"csv={csv_path}", flush=True)


if __name__ == "__main__":
    main()
