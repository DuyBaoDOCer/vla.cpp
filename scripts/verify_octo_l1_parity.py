#!/usr/bin/env python3
"""TIP-06: compare octo_l1_parity_dump's T0-T5 C++ activations against the OctoPt stagewise
golden dump (octo-pytorch-kamusarj's scripts/dump_l1_stagewise_golden.py). Mirrors
verify_octo_parity.py's .npy/.f32 parsing (no numpy dependency -- ctest's Python3_EXECUTABLE
may be a bare interpreter) but is deliberately standalone/simpler: this golden format has no
dataset-name wrapper or boundary-renaming map, since both sides were designed together and
use the SAME stage names (t0.proprio_normalized, t0.proprio_tokens, t1t2.readout_action,
t3.map_attn_out, t3.map_emb, t4.mean_normalized, t5.action_unnormalized).
"""

from __future__ import annotations

import argparse
import ast
import array
import math
from pathlib import Path

STAGES = (
    "t0.proprio_normalized",
    "t0.proprio_tokens",
    "t1t2.readout_action",
    "t3.map_attn_out",
    "t3.map_emb",
    "t4.mean_normalized",
    "t5.action_unnormalized",
)


def load_dump_manifest(path: Path) -> dict[str, tuple[Path, str, tuple[int, ...]]]:
    rows: dict[str, tuple[Path, str, tuple[int, ...]]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4 or parts[2] != "float32":
            raise ValueError(f"bad dump manifest line: {raw}")
        rows[parts[0]] = (Path(parts[1]), parts[2], tuple(int(x) for x in parts[3:]))
    return rows


def load_npy_float32(path: Path) -> tuple[tuple[int, ...], array.array]:
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:6] != b"\x93NUMPY":
        raise ValueError(f"not a .npy file: {path}")
    major = raw[6]
    pos = 8
    if major == 1:
        hlen = int.from_bytes(raw[pos : pos + 2], "little")
        pos += 2
    elif major in (2, 3):
        hlen = int.from_bytes(raw[pos : pos + 4], "little")
        pos += 4
    else:
        raise ValueError(f"unsupported .npy version {major}: {path}")
    header = raw[pos : pos + hlen].decode("latin1").strip()
    pos += hlen
    meta = ast.literal_eval(header)
    if meta.get("descr") not in ("<f4", "|f4"):
        raise ValueError(f"expected float32 .npy, got {meta.get('descr')}: {path}")
    if meta.get("fortran_order"):
        raise ValueError(f"expected C-order .npy: {path}")
    shape = tuple(int(x) for x in meta["shape"])
    n = math.prod(shape) if shape else 1
    data = array.array("f")
    data.frombytes(raw[pos : pos + n * 4])
    if len(data) != n:
        raise ValueError(f"truncated .npy payload: {path}")
    return shape, data


def load_raw_float32(path: Path, n: int) -> array.array:
    data = array.array("f")
    with path.open("rb") as f:
        data.fromfile(f, n)
    if len(data) != n:
        raise ValueError(f"truncated dump payload: {path}")
    return data


def max_abs_diff(a: array.array, b: array.array) -> float:
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, type=Path, help="ep0_t<N> dir from dump_l1_stagewise_golden.py")
    ap.add_argument("--dump", required=True, type=Path, help="dump dir containing manifest.txt (octo_l1_parity_dump --out)")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    dump = load_dump_manifest(args.dump / "manifest.txt")

    print("stage\tgolden_shape\tdump_shape\tmax_abs_diff\tstatus")
    ok = True
    for name in STAGES:
        golden_path = args.golden / f"{name}.npy"
        if not golden_path.exists():
            raise SystemExit(f"missing golden file: {golden_path}")
        if name not in dump:
            raise SystemExit(f"missing dump boundary: {name} (dump manifest={sorted(dump)})")
        golden_shape, golden_data = load_npy_float32(golden_path)
        dump_path, _, dump_shape = dump[name]
        dump_n = math.prod(dump_shape) if dump_shape else 1
        golden_n = math.prod(golden_shape) if golden_shape else 1
        if dump_n != golden_n:
            raise SystemExit(
                f"{name}: element count mismatch dump={dump_shape}({dump_n}) golden={golden_shape}({golden_n})")
        # manifest.txt already stores a directly-usable path (write_f32_dump wrote
        # "<dump_dir>/<name>.f32" verbatim) -- use it as-is, don't re-prefix with --dump.
        dump_data = load_raw_float32(dump_path, dump_n)
        diff = max_abs_diff(dump_data, golden_data)
        status = "PASS" if diff < args.tol else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{name}\t{golden_shape}\t{dump_shape}\t{diff:.8f}\t{status}")

    if not ok:
        print(f"FAILED: one or more stages exceeded tol={args.tol}")
        return 1
    print(f"PASS: all {len(STAGES)} stages < tol={args.tol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
