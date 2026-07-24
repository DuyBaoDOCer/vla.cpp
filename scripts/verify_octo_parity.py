#!/usr/bin/env python3
"""Compare Octo tokenizer boundary dumps against npy+manifest golden traces."""

from __future__ import annotations

import argparse
import ast
import array
import json
import math
from pathlib import Path


BOUNDARY_MAP = {
    "obs.primary.tok": "octo_transformer.obs_primary.tokens_after_tokenizer",
    "obs.primary.proj": "octo_transformer.obs_primary.tokens_after_projection",
    "obs.primary.pos": "octo_transformer.obs_primary.tokens_after_pos_embedding",
    "obs.wrist.tok": "octo_transformer.obs_wrist.tokens_after_tokenizer",
    "obs.wrist.proj": "octo_transformer.obs_wrist.tokens_after_projection",
    "obs.wrist.pos": "octo_transformer.obs_wrist.tokens_after_pos_embedding",
}

LANGUAGE_BOUNDARY_MAP = {
    "lang.proj": "octo_transformer.task_language.tokens_after_projection",
    "lang.pos": "octo_transformer.task_language.tokens_after_pos_embedding",
    "repeated_language": "octo_transformer.obs_task_language.tokens_repeated_task",
}


def load_dump_manifest(path: Path) -> dict[str, tuple[Path, tuple[int, ...]]]:
    rows: dict[str, tuple[Path, tuple[int, ...]]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4 or parts[2] != "float32":
            raise ValueError(f"bad dump manifest line: {raw}")
        rows[parts[0]] = (Path(parts[1]), tuple(int(x) for x in parts[3:]))
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
    n = math.prod(shape)
    data = array.array("f")
    data.frombytes(raw[pos : pos + n * 4])
    if len(data) != n:
        raise ValueError(f"truncated .npy payload: {path}")
    return shape, data


def load_raw_float32(path: Path, shape: tuple[int, ...]) -> array.array:
    n = math.prod(shape)
    data = array.array("f")
    with path.open("rb") as f:
        data.fromfile(f, n)
    if len(data) != n:
        raise ValueError(f"truncated dump payload: {path}")
    return data


def stats(actual: array.array, golden: array.array) -> tuple[float, float, float]:
    max_abs = 0.0
    ref_max = 0.0
    dot = 0.0
    aa = 0.0
    bb = 0.0
    for a, b in zip(actual, golden):
        af = float(a)
        bf = float(b)
        diff = abs(af - bf)
        max_abs = max(max_abs, diff)
        ref_max = max(ref_max, abs(bf))
        dot += af * bf
        aa += af * af
        bb += bf * bf
    max_rel = max_abs / max(ref_max, 1e-12)
    cos = dot / (math.sqrt(aa) * math.sqrt(bb)) if aa and bb else 1.0
    return max_abs, max_rel, cos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, type=Path, help="golden case directory containing manifest.json")
    ap.add_argument("--dump", required=True, type=Path, help="VLA_OCTO_DUMP directory containing manifest.txt")
    ap.add_argument("--oracle-dump", type=Path, help="optional prior oracle dump directory containing manifest.txt")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--oracle-tol", type=float, default=1e-4)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    golden_manifest = json.loads((args.golden / "manifest.json").read_text(encoding="utf-8"))
    if golden_manifest.get("format") != "npy+manifest.v1":
        raise SystemExit(f"unsupported golden format: {golden_manifest.get('format')}")
    if golden_manifest.get("layout") != "row-major":
        raise SystemExit(f"unsupported golden layout: {golden_manifest.get('layout')}")
    tensors = golden_manifest["tensors"]
    dump = load_dump_manifest(args.dump / "manifest.txt")
    oracle = load_dump_manifest(args.oracle_dump / "manifest.txt") if args.oracle_dump else None

    rows = []
    ok = True
    print("boundary\tgolden\tshape\tmax_abs_err\tmax_rel_err\tcosine\tstatus\toracle_max_rel_err\toracle_status")
    boundary_map = dict(BOUNDARY_MAP)
    if any(name in dump for name in LANGUAGE_BOUNDARY_MAP):
        boundary_map.update(LANGUAGE_BOUNDARY_MAP)

    for dump_name, golden_name in boundary_map.items():
        if dump_name not in dump:
            raise SystemExit(f"missing dump boundary: {dump_name}")
        if golden_name not in tensors:
            raise SystemExit(f"missing golden boundary: {golden_name}")
        dump_path, dump_shape = dump[dump_name]
        ginfo = tensors[golden_name]
        golden_shape, golden = load_npy_float32(args.golden / ginfo["file"])
        actual = load_raw_float32(dump_path, dump_shape)
        if golden_shape != dump_shape:
            raise SystemExit(f"shape mismatch {dump_name}: dump={dump_shape} golden={golden_shape}")
        max_abs, max_rel, cos = stats(actual, golden)
        status = "PASS" if max_rel <= args.tol else "FAIL"
        oracle_max_rel = None
        oracle_status = "SKIP"
        if oracle is not None:
            if dump_name not in oracle:
                raise SystemExit(f"missing oracle boundary: {dump_name}")
            oracle_path, oracle_shape = oracle[dump_name]
            if oracle_shape != dump_shape:
                raise SystemExit(f"oracle shape mismatch {dump_name}: dump={dump_shape} oracle={oracle_shape}")
            oracle_data = load_raw_float32(oracle_path, oracle_shape)
            _, oracle_max_rel, _ = stats(actual, oracle_data)
            oracle_status = "PASS" if oracle_max_rel <= args.oracle_tol else "FAIL"
        ok = ok and status == "PASS" and oracle_status != "FAIL"
        row = {
            "boundary": dump_name,
            "golden": golden_name,
            "shape": list(dump_shape),
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "cosine": cos,
            "status": status,
            "oracle_max_rel_err": oracle_max_rel,
            "oracle_status": oracle_status,
        }
        rows.append(row)
        oracle_rel_s = "" if oracle_max_rel is None else f"{oracle_max_rel:.6g}"
        print(f"{dump_name}\t{golden_name}\t{list(dump_shape)}\t{max_abs:.6g}\t{max_rel:.6g}\t{cos:.9f}\t{status}\t{oracle_rel_s}\t{oracle_status}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"ok": ok, "tol": args.tol, "oracle_tol": args.oracle_tol, "rows": rows}, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
