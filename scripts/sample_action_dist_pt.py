#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""TIP-009: OctoPt reference sampler for statistical action-distribution parity.

Loads a golden case's OWN saved observation/task tensors directly (dataset-agnostic:
whatever input.observation.*/input.task.* keys exist in the case's manifest.json are
loaded, none are assumed present -- e.g. bridge_debug has no
input.task.image_primary/wrist at all, since it's a real dataset with a language-only
task space; those keys are simply absent from `tasks`, matching what OctoPt's own
dump_octo_golden_trace_bridge_debug_pt.py passes to sample_actions() for that case).

Calls model.sample_actions(...) n_samples times with fresh per-sample noise (a
torch.Generator seeded seed+i each call, matching vla.cpp's std::mt19937(seed+i)),
collecting the normalized [N,4,7] action and the [N,2,28] initial DDPM noise actually
consumed each call (captured via a minimal trace shim -- DiffusionActionHeadPt.
predict_action already calls trace.write("action_head.predict_action.initial_noise",
current_x) unconditionally when a trace object is passed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def _load_tensor(case_dir: Path, manifest: dict, key: str) -> torch.Tensor:
    info = manifest["tensors"][key]
    arr = np.load(case_dir / info["file"])
    return torch.from_numpy(arr)


def _build_tree(case_dir: Path, manifest: dict, prefix: str) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    full_prefix = prefix + "."
    for key in manifest["tensors"]:
        if not key.startswith(full_prefix):
            continue
        rel = key[len(full_prefix):]
        parts = rel.split(".")
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _load_tensor(case_dir, manifest, key)
    return tree


def _to_device(tree, device: str):
    if isinstance(tree, dict):
        return {k: _to_device(v, device) for k, v in tree.items()}
    return tree.to(device)


class NoiseCapture:
    """Duck-typed trace shim. DiffusionActionHeadPt.predict_action calls
    trace.write(name, value) for many boundaries during sampling; we keep only the
    one we need and no-op everything else (including write_tree, called from the
    transformer forward pass for unrelated boundaries).
    """

    def __init__(self) -> None:
        self.initial_noise: Optional[torch.Tensor] = None

    def write(self, name: str, value) -> None:
        if name == "action_head.predict_action.initial_noise":
            self.initial_noise = value.detach().cpu().clone()

    def write_tree(self, *_args, **_kwargs) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="OctoPt free-sample reference for TIP-009 distribution parity.")
    ap.add_argument("--case", required=True, type=Path, help="golden case directory (contains manifest.json + tensors/)")
    ap.add_argument("--checkpoint", default="hf://rail-berkeley/octo-small-1.5")
    ap.add_argument("--octo-root", type=Path, default=Path(__file__).resolve().parents[1] / "octo-pytorch")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-samples", type=Path, default=Path("pt_samples.npy"))
    ap.add_argument("--out-noise", type=Path, default=Path("pt_noise.npy"))
    args = ap.parse_args()

    if args.octo_root.exists():
        sys.path.insert(0, str(args.octo_root))
    from octo.model.octo_model_pt import OctoModelPt  # noqa: E402 (needs sys.path insert first)

    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    manifest = json.loads((args.case / "manifest.json").read_text(encoding="utf-8"))

    observations = _to_device(_build_tree(args.case, manifest, "input.observation"), args.device)
    tasks = _to_device(_build_tree(args.case, manifest, "input.task"), args.device)
    print(f"observation keys: {sorted(observations.keys())}")
    print(f"task keys: {sorted(tasks.keys())}")
    if "image_primary" not in tasks:
        print("note: no input.task.image_primary in this case (language-only task space) "
              "-- passing tasks dict as-is, matching how this case's own golden trace was generated")

    timestep_pad_mask = observations["timestep_pad_mask"]

    print(f"loading {args.checkpoint} on {args.device} ...")
    loaded = OctoModelPt.load_pretrained_from_jax(args.checkpoint, skip_keys_regex=".*hf_model")
    model = loaded["octo_model"].to(args.device).eval()

    samples = np.zeros((args.n_samples, 4, 7), dtype=np.float32)
    noise = np.zeros((args.n_samples, 2, 28), dtype=np.float32)

    with torch.no_grad():
        for i in range(args.n_samples):
            torch.manual_seed(args.seed + i)  # belt-and-suspenders; sampling itself is generator-scoped
            cap = NoiseCapture()
            gen = torch.Generator(device=args.device).manual_seed(args.seed + i)
            action = model.sample_actions(
                observations,
                tasks,
                timestep_pad_mask=timestep_pad_mask,
                train=False,
                generator=gen,
                trace=cap,
            )
            samples[i] = action.detach().cpu().numpy().reshape(4, 7)
            if cap.initial_noise is None:
                raise RuntimeError("trace shim did not capture initial_noise -- predict_action's trace contract changed")
            noise[i] = cap.initial_noise.numpy().reshape(2, 28)
            if (i + 1) % 10 == 0 or i == args.n_samples - 1:
                print(f"  sample {i + 1}/{args.n_samples}")

    args.out_samples.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_samples, samples)
    np.save(args.out_noise, noise)
    print(f"wrote {args.out_samples} {samples.shape}, {args.out_noise} {noise.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
