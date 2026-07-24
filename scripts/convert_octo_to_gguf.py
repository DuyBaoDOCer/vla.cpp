#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import gguf

ARCH = "octo"
MODEL_ID = "hf://rail-berkeley/octo-small-1.5"

OCTO_META: dict[str, Any] = {
    "architecture": "octo-small-1.5",
    "embedding_length": 384,
    "block_count": 12,
    "attention.head_count": 6,
    "feed_forward_length": 1536,
    "attention.layer_norm_eps": 1e-6,
    "window_size": 2,
    "action.horizon": 4,
    "action.dim": 7,
    "readout.count": 1,
    "tokens.primary": 256,
    "tokens.wrist": 64,
    "tokens.language": 16,
    "image.primary_size": 256,
    "image.wrist_size": 128,
    "diffusion.steps": 20,
    "diffusion.beta_schedule": "cosine",
    "diffusion.s": 0.008,
    "diffusion.max_action": 5,
    "diffusion.time_dim": 32,
    "diffusion.hidden": 256,
    "diffusion.num_blocks": 3,
}

IGNORED_PATTERNS = (
    re.compile(r"module\.octo_transformer\.task_tokenizers\.language\.hf_model\."),
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"cannot JSON encode {type(obj).__name__}")


def _add_meta(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    full = f"octo.{key}"
    if isinstance(value, str):
        writer.add_string(full, value)
    elif isinstance(value, bool):
        writer.add_bool(full, value)
    elif isinstance(value, int):
        writer.add_uint32(full, value)
    elif isinstance(value, float):
        writer.add_float32(full, value)
    else:
        raise TypeError(f"unsupported metadata {full}={value!r}")


def _f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()


def _strip_prefix(key: str) -> str:
    return key.removeprefix("module.")


def map_key(pt_key: str) -> str | None:
    k = _strip_prefix(pt_key)

    m = re.fullmatch(r"octo_transformer\.observation_tokenizers\.(primary|wrist)\.encoder_def\.layers\.(\d+)\.0\.(weight|bias)", k)
    if m:
        view, idx, leaf = m.groups()
        return f"octo.obs.{view}.stem.{idx}.conv.{leaf}"

    m = re.fullmatch(r"octo_transformer\.observation_tokenizers\.(primary|wrist)\.encoder_def\.layers\.(\d+)\.1\.(weight|bias)", k)
    if m:
        view, idx, leaf = m.groups()
        return f"octo.obs.{view}.stem.{idx}.gn.{leaf}"

    m = re.fullmatch(r"octo_transformer\.observation_tokenizers\.(primary|wrist)\.encoder_def\.embedding\.(weight|bias)", k)
    if m:
        view, leaf = m.groups()
        return f"octo.obs.{view}.patch_embd.{leaf}"

    m = re.fullmatch(r"octo_transformer\.obs_projections\.obs_(primary|wrist)_projection\.(weight|bias)", k)
    if m:
        view, leaf = m.groups()
        return f"octo.obs.{view}.proj.{leaf}"

    m = re.fullmatch(r"octo_transformer\.obs_(primary|wrist)_pos_embedding", k)
    if m:
        return f"octo.obs.{m.group(1)}.pos_embd"

    m = re.fullmatch(r"octo_transformer\.task_projections\.task_language_projection\.(weight|bias)", k)
    if m:
        return f"octo.task.language.proj.{m.group(1)}"
    if k == "octo_transformer.task_language_pos_embedding":
        return "octo.task.language.pos_embd"
    if k == "octo_transformer.readout_action_pos_embedding":
        return "octo.readout.action.pos_embd"

    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\.(\d+)\.layer_norm1\.(weight|bias)", k)
    if m:
        return f"octo.blk.{m.group(1)}.attn_norm.{m.group(2)}"
    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\.(\d+)\.self_attention\.in_proj_(weight|bias)", k)
    if m:
        return f"octo.blk.{m.group(1)}.attn_qkv.{m.group(2)}"
    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\.(\d+)\.self_attention\.out_proj\.(weight|bias)", k)
    if m:
        return f"octo.blk.{m.group(1)}.attn_o.{m.group(2)}"
    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\.(\d+)\.layer_norm2\.(weight|bias)", k)
    if m:
        return f"octo.blk.{m.group(1)}.ffn_norm.{m.group(2)}"
    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\.(\d+)\.mlp_block\.dense1\.(weight|bias)", k)
    if m:
        return f"octo.blk.{m.group(1)}.ffn_up.{m.group(2)}"
    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\.(\d+)\.mlp_block\.dense2\.(weight|bias)", k)
    if m:
        return f"octo.blk.{m.group(1)}.ffn_down.{m.group(2)}"
    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.layer_norm\.(weight|bias)", k)
    if m:
        return f"octo.output_norm.{m.group(1)}"

    p = "heads.action.diffusion_model."
    if k == p + "time_preprocess.w":
        return "octo.head.diffusion.time_fourier.weight"
    m = re.fullmatch(re.escape(p) + r"cond_encoder\.layers\.(0|2)\.(weight|bias)", k)
    if m:
        idx = "0" if m.group(1) == "0" else "1"
        return f"octo.head.diffusion.cond.{idx}.{m.group(2)}"
    m = re.fullmatch(re.escape(p) + r"reverse_network\.linear1\.(weight|bias)", k)
    if m:
        return f"octo.head.diffusion.reverse.in.{m.group(1)}"
    m = re.fullmatch(re.escape(p) + r"reverse_network\.blocks\.(\d+)\.layer_norm\.(weight|bias)", k)
    if m:
        return f"octo.head.diffusion.reverse.blk.{m.group(1)}.ln.{m.group(2)}"
    m = re.fullmatch(re.escape(p) + r"reverse_network\.blocks\.(\d+)\.linear1\.(weight|bias)", k)
    if m:
        return f"octo.head.diffusion.reverse.blk.{m.group(1)}.fc1.{m.group(2)}"
    m = re.fullmatch(re.escape(p) + r"reverse_network\.blocks\.(\d+)\.linear2\.(weight|bias)", k)
    if m:
        return f"octo.head.diffusion.reverse.blk.{m.group(1)}.fc2.{m.group(2)}"
    m = re.fullmatch(re.escape(p) + r"reverse_network\.linear2\.(weight|bias)", k)
    if m:
        return f"octo.head.diffusion.reverse.out.{m.group(1)}"

    return None


def _validate_required(mapped: dict[str, str]) -> list[str]:
    required: list[str] = []
    for view in ("primary", "wrist"):
        for i in range(4):
            for leaf in ("weight", "bias"):
                required.append(f"octo.obs.{view}.stem.{i}.conv.{leaf}")
                required.append(f"octo.obs.{view}.stem.{i}.gn.{leaf}")
        for leaf in ("weight", "bias"):
            required.append(f"octo.obs.{view}.patch_embd.{leaf}")
            required.append(f"octo.obs.{view}.proj.{leaf}")
        required.append(f"octo.obs.{view}.pos_embd")
    required += ["octo.task.language.proj.weight", "octo.task.language.proj.bias", "octo.task.language.pos_embd", "octo.readout.action.pos_embd"]
    for i in range(12):
        for stem in ("attn_norm", "attn_qkv", "attn_o", "ffn_norm", "ffn_up", "ffn_down"):
            for leaf in ("weight", "bias"):
                required.append(f"octo.blk.{i}.{stem}.{leaf}")
    required += ["octo.head.diffusion.time_fourier.weight"]
    for i in range(2):
        for leaf in ("weight", "bias"):
            required.append(f"octo.head.diffusion.cond.{i}.{leaf}")
    for leaf in ("weight", "bias"):
        required.append(f"octo.head.diffusion.reverse.in.{leaf}")
        required.append(f"octo.head.diffusion.reverse.out.{leaf}")
    for i in range(3):
        for sub in ("ln", "fc1", "fc2"):
            for leaf in ("weight", "bias"):
                required.append(f"octo.head.diffusion.reverse.blk.{i}.{sub}.{leaf}")
    have = set(mapped.values())
    return [k for k in required if k not in have]


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Octo PyTorch state_dict to F32 GGUF.")
    ap.add_argument("--out", type=Path, default=Path("octo-small-1.5-f32.gguf"))
    ap.add_argument("--octo-root", type=Path, default=Path(__file__).resolve().parents[1] / "octo-pytorch")
    ap.add_argument("--allow-unmapped", action="store_true", help="write known mapped tensors and report unmapped keys instead of failing")
    args = ap.parse_args()

    if args.octo_root.exists():
        sys.path.insert(0, str(args.octo_root))

    from octo.model.octo_model_pt import OctoModelPt

    print(f"loading {MODEL_ID} via OctoModelPt.load_pretrained_from_jax ...")
    loaded = OctoModelPt.load_pretrained_from_jax(MODEL_ID, skip_keys_regex=".*hf_model")
    m = loaded["octo_model"]
    sd = m.state_dict()

    print("state_dict keys and shapes:")
    for key in sorted(sd):
        print(f"{key}\t{tuple(sd[key].shape)}\t{sd[key].dtype}")

    mapped: dict[str, str] = {}
    ignored: list[str] = []
    unmapped: list[str] = []
    for key, tensor in sd.items():
        if not tensor.is_floating_point():
            continue
        if any(pattern.match(key) for pattern in IGNORED_PATTERNS):
            ignored.append(key)
            continue
        dst = map_key(key)
        if dst is None:
            unmapped.append(key)
        elif dst in mapped.values():
            raise SystemExit(f"duplicate GGUF destination {dst} from {key}")
        else:
            mapped[key] = dst

    missing = _validate_required(mapped)
    if ignored:
        print("IGNORED STATE_DICT KEYS:")
        for key in sorted(ignored):
            print(f"  {key} {tuple(sd[key].shape)}")

    if unmapped or missing:
        print("TENSOR MAP REPORT:")
        if unmapped:
            print("unmapped state_dict keys:")
            for key in sorted(unmapped):
                print(f"  {key} {tuple(sd[key].shape)}")
        if missing:
            print("missing required GGUF tensors:")
            for key in missing:
                print(f"  {key}")
        if unmapped or missing:
            if not args.allow_unmapped:
                raise SystemExit("Octo tensor map is incomplete; re-run with --allow-unmapped only for investigation")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.out), arch=ARCH)
    for key, value in OCTO_META.items():
        _add_meta(writer, key, value)
    writer.add_string("octo.dataset_statistics", json.dumps(m.dataset_statistics, default=_json_default, sort_keys=True))

    rows = []
    for src, dst in sorted(mapped.items(), key=lambda kv: kv[1]):
        tensor = sd[src]
        writer.add_tensor(dst, _f32(tensor), raw_dtype=gguf.GGMLQuantizationType.F32)
        rows.append({"state_dict": src, "gguf": dst, "shape": list(tensor.shape)})
        print(f"map {src} {tuple(tensor.shape)} -> {dst}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    report = args.out.with_suffix(args.out.suffix + ".tensor_map.json")
    report.write_text(json.dumps({"mapped": rows, "ignored": ignored, "unmapped": unmapped, "missing_required": missing}, indent=2), encoding="utf-8")
    print(f"done: {args.out} ({args.out.stat().st_size / (1024 * 1024):.1f} MiB)")
    print(f"tensor map: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
