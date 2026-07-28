#!/usr/bin/env python3
"""Add zero-filled wrist placeholder tensors to single-camera LIBERO golden cases.

octo_parity_dump's loader (octo_dump_tokenizer_case in src/models/octo.cpp) hard-requires
tensors/input.observation.image_wrist.npy and .../pad_mask_dict.image_wrist.npy to exist in
any golden case directory it reads (unlike the task-side wrist tensor, which has a zero-fill
fallback keyed off the observation tensor's own shape). The cyrusneary LIBERO checkpoint
(TIP-GOLD) is genuinely single-camera -- its golden traces never ran a wrist observation
through the model at all, so these files don't exist there.

This script adds all-zero / all-False placeholders so the C++ loader can read the case
directory without crashing. It does not touch any tensor that was actually produced by the
model (Tier B trace values are untouched). The wrist-derived boundaries this unblocks
(obs.wrist.*, bt.obs_wrist, bt.input, bt.mask, bt.output) are excluded from parity comparison
by verify_octo_parity.py's --exclude flag for LIBERO cases -- see TIP-HARNESS report for why.

Idempotent: skips files that already exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WRIST_IMAGE_SIZE = 128  # matches the bridge golden's own input.observation.image_wrist.npy


def patch_case(case_dir: Path) -> list[str]:
    tensors_dir = case_dir / "tensors"
    primary_path = tensors_dir / "input.observation.image_primary.npy"
    if not primary_path.exists():
        raise SystemExit(f"missing {primary_path}; not a valid golden case dir")
    primary = np.load(primary_path)
    if primary.ndim != 5:
        raise SystemExit(f"expected 5D image_primary [B,T,C,H,W], got {primary.shape}")
    batch, window = primary.shape[0], primary.shape[1]

    written = []
    wrist_img_path = tensors_dir / "input.observation.image_wrist.npy"
    if not wrist_img_path.exists():
        wrist = np.zeros((batch, window, 3, WRIST_IMAGE_SIZE, WRIST_IMAGE_SIZE), dtype=np.uint8)
        np.save(wrist_img_path, wrist)
        written.append(str(wrist_img_path))

    wrist_mask_path = tensors_dir / "input.observation.pad_mask_dict.image_wrist.npy"
    if not wrist_mask_path.exists():
        mask = np.zeros((batch, window), dtype=bool)
        np.save(wrist_mask_path, mask)
        written.append(str(wrist_mask_path))

    manifest_path = case_dir / "manifest.json"
    if manifest_path.exists() and written:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        note = manifest.setdefault("metadata", {}).setdefault("harness_patch", {})
        note["patched_wrist_placeholder"] = True
        note["reason"] = (
            "checkpoint is single-camera; placeholder added only so octo_parity_dump's "
            "loader can read the case dir, excluded from comparison via --exclude"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("golden_dir", type=Path, help="LIBERO golden root (contains <case>/ subdirs)")
    ap.add_argument("--case", action="append", default=[], help="case name(s); default: all subdirs")
    args = ap.parse_args()

    cases = args.case or [p.name for p in args.golden_dir.iterdir() if (p / "tensors").is_dir()]
    for case in cases:
        written = patch_case(args.golden_dir / case)
        if written:
            print(f"{case}: wrote {written}")
        else:
            print(f"{case}: already patched (nothing to do)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
