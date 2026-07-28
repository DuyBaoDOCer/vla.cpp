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

"""Tier-A parity tests for Octo's client-side image preprocessing (rotate180 + resize
to 256) -- eval/client/adapters.py:octo_preprocess_image. Ref: TIP-P.

octo_preprocess_image() is pure numpy/PIL, but the module it lives in
(eval/client/adapters.py) unconditionally imports torch/tree/lerobot at module scope
for LeRobotPipelineAdapter. Those aren't needed here and may not be installed outside
the LIBERO eval venv, so they're stubbed before import (same technique as
test_converters.py's `gguf` stub) to keep this test runnable with just numpy+PIL.

Golden tier-A cases (raw.npy/model_entry.npy pairs) are not committed to the repo, so
the parity tests are gated on VLA_OCTO_LIBERO_GOLDEN_DIR (same env var CMake's
octo_parity_libero_* ctest cases use) and skipped/no-op if it isn't set. The rotate
recovery test needs no golden data and always runs.
"""

import importlib.util
import os
import pathlib
import sys
import types

import numpy as np


def _load_adapters():
    torch_stub = sys.modules.setdefault("torch", types.ModuleType("torch"))
    if not hasattr(torch_stub, "from_numpy"):
        torch_stub.from_numpy = object()
    tree_stub = sys.modules.setdefault("tree", types.ModuleType("tree"))
    if not hasattr(tree_stub, "map_structure"):
        tree_stub.map_structure = object()
    lerobot = sys.modules.setdefault("lerobot", types.ModuleType("lerobot"))
    for sub, attrs in {
        "lerobot.envs.utils": ["preprocess_observation"],
        "lerobot.processor.env_processor": ["LiberoProcessorStep"],
        "lerobot.processor.pipeline": ["PolicyProcessorPipeline"],
        "lerobot.utils.constants": ["ACTION"],
    }.items():
        mod = sys.modules.setdefault(sub, types.ModuleType(sub))
        for attr in attrs:
            if not hasattr(mod, attr):
                setattr(mod, attr, object())
    del lerobot

    path = pathlib.Path(__file__).resolve().parents[2] / "eval" / "client" / "adapters.py"
    spec = importlib.util.spec_from_file_location("octo_tipp_adapters", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Per-case tier-A tolerance (uint8 max_abs vs golden model_entry.npy). synthetic_zero is
# an all-black frame -> JPEG is lossless there, so rotate+resize must be bit-exact
# (max_abs=0). The other two carry real JPEG round-trip noise in golden itself
# (max_abs_diff_vs_source_frame = 2 / 20 respectively, per
# libero_golden_summary.json) -- <=25 is soft slack for that, not a license to get the
# rotation/resize wrong: a wrong rotation or an axis-swapped resize produces max_abs in
# the 127-255 range (near-uncorrelated pixels), which this tolerance would still catch.
TIER_A_TOLERANCE = {
    "synthetic_zero": 0,
    "synthetic_ramp": 25,
    "example_frame": 25,
}


def test_octo_rotate_recovers_frame():
    """Step 3: preprocess(frame[::-1,::-1]) must recover frame exactly (no JPEG in the
    loop here, so this isolates just the rotate+identity-resize logic)."""
    adapters = _load_adapters()
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

    upside_down = frame[::-1, ::-1]
    recovered = adapters.octo_preprocess_image(upside_down, image_size=256)

    assert recovered.shape == frame.shape
    assert recovered.dtype == np.uint8
    diff = np.abs(recovered.astype(np.int32) - frame.astype(np.int32))
    assert diff.max() == 0, f"rotate180 did not recover the original frame: max_abs={diff.max()}"


def _tier_a_case(adapters, golden_dir: pathlib.Path, case: str) -> tuple[int, float]:
    case_dir = golden_dir / case / "preprocessing"
    raw = np.load(case_dir / "raw.npy")
    model_entry = np.load(case_dir / "model_entry.npy")

    got = adapters.octo_preprocess_image(raw, image_size=256)
    assert got.shape == model_entry.shape, f"{case}: shape {got.shape} != golden {model_entry.shape}"

    diff = np.abs(got.astype(np.int32) - model_entry.astype(np.int32))
    return int(diff.max()), float(diff.mean())


def _run_tier_a(case: str):
    golden_dir_env = os.environ.get("VLA_OCTO_LIBERO_GOLDEN_DIR")
    if not golden_dir_env:
        return  # gated: no golden dir configured, nothing to verify (matches ctest gating)
    golden_dir = pathlib.Path(golden_dir_env)
    case_dir = golden_dir / case / "preprocessing"
    if not (case_dir / "raw.npy").exists():
        return

    adapters = _load_adapters()
    max_abs, mean_abs = _tier_a_case(adapters, golden_dir, case)
    tol = TIER_A_TOLERANCE[case]
    assert max_abs <= tol, (
        f"{case}: max_abs={max_abs} exceeds tolerance {tol} vs golden model_entry.npy "
        f"(mean_abs={mean_abs:.4f}) -- check rotation direction/axis if this is large "
        f"(>~30), not just JPEG noise"
    )
    print(f"tier_a[{case}]: max_abs={max_abs} mean_abs={mean_abs:.4f} tol={tol} PASS")


def test_octo_tier_a_synthetic_zero():
    _run_tier_a("synthetic_zero")


def test_octo_tier_a_synthetic_ramp():
    _run_tier_a("synthetic_ramp")


def test_octo_tier_a_example_frame():
    _run_tier_a("example_frame")


if __name__ == "__main__":
    test_octo_rotate_recovers_frame()
    print("test_octo_rotate_recovers_frame: OK")
    for case in TIER_A_TOLERANCE:
        _run_tier_a(case)
    print("test_octo_preprocessing: done")
