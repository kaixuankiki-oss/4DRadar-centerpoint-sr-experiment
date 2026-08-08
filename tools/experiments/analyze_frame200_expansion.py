#!/usr/bin/env python3
"""Audit label-independent raw-return expansion rules on Frame200.

Annotations are used only to measure candidate precision after a rule has been
applied.  The rules themselves depend exclusively on raw point position, RCS,
AbsV, range, and raw voxel occupancy, matching reconstructed_inference.py.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


PCD_TYPES = {
    ("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1",
    ("U", 2): "<u2", ("U", 4): "<u4", ("I", 1): "i1",
    ("I", 2): "<i2", ("I", 4): "<i4",
}
CLASS_MAP = {
    "Car": "Car", "Truck": "LargeVehicle", "Bus": "LargeVehicle",
    "Split_vehicle": "LargeVehicle", "Tricycle": "Cyclist",
    "Cyclist": "Cyclist", "Pedestrian": "Pedestrian",
}
OFFSET_PATTERNS = {
    "long1": ((-1, 0), (1, 0)),
    "long2": ((-2, 0), (-1, 0), (1, 0), (2, 0)),
    "cross1": ((-1, 0), (1, 0), (0, -1), (0, 1)),
    "box1": tuple(
        (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0)
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--raw-val", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--min-abs-v", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--min-rcs", type=float, nargs="+", default=[0.0, 2.0, 5.0, 10.0])
    parser.add_argument("--patterns", nargs="+", choices=sorted(OFFSET_PATTERNS),
                        default=["long1", "long2", "cross1", "box1"])
    parser.add_argument("--max-range", type=float, default=50.0)
    parser.add_argument("--voxel-size", type=float, nargs=2, default=[0.25, 0.20])
    return parser.parse_args()


def read_pcd(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD has no DATA line: {path}")
            parts = line.decode("utf-8", errors="replace").strip().split()
            if parts and not parts[0].startswith("#"):
                header[parts[0]] = parts[1:]
            if parts and parts[0] == "DATA":
                break
        if header["DATA"][0] != "binary":
            raise ValueError(f"Only binary PCD is supported: {path}")
        fields = header["FIELDS"]
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        dtype = np.dtype([
            (name, PCD_TYPES[(kind, size)]) if count == 1
            else (name, PCD_TYPES[(kind, size)], (count,))
            for name, size, kind, count in zip(fields, sizes, types, counts)
        ])
        return np.fromfile(stream, dtype=dtype, count=int(header["POINTS"][0]))


def transform_xyz(xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float32)], axis=1)
    return (xyz_h @ np.asarray(matrix, dtype=np.float32).T)[:, :3]


def points_in_boxes(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return a [num_points, num_boxes] inclusion mask for [x,y,z,dx,dy,dz,yaw]."""
    if len(points) == 0 or len(boxes) == 0:
        return np.zeros((len(points), len(boxes)), dtype=bool)
    delta = points[:, None, :3] - boxes[None, :, :3]
    heading = boxes[:, 6]
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    local_x = delta[:, :, 0] * cos_h + delta[:, :, 1] * sin_h
    local_y = -delta[:, :, 0] * sin_h + delta[:, :, 1] * cos_h
    return (
        (np.abs(local_x) <= boxes[None, :, 3] / 2)
        & (np.abs(local_y) <= boxes[None, :, 4] / 2)
        & (np.abs(delta[:, :, 2]) <= boxes[None, :, 5] / 2)
    )


def expand(points: np.ndarray, min_abs_v: float, min_rcs: float, max_range: float,
           voxel_size: tuple[float, float], offsets: tuple[tuple[int, int], ...]):
    vx, vy = voxel_size
    x = points["x"].astype(np.float32)
    y = points["y"].astype(np.float32)
    z = points["z"].astype(np.float32)
    ranges = points["range"].astype(np.float32) if "range" in points.dtype.names \
        else np.sqrt(x * x + y * y + z * z)
    source_voxels = np.stack([np.floor(x / vx), np.floor(y / vy)], axis=1).astype(np.int32)
    occupied = {tuple(item) for item in source_voxels}
    seed_mask = (
        (ranges < max_range)
        & (np.abs(points["AbsV"].astype(np.float32)) >= min_abs_v)
        & (points["RCS"].astype(np.float32) >= min_rcs)
    )
    best = {}
    for index in np.flatnonzero(seed_mask):
        source = source_voxels[index]
        for dx, dy in offsets:
            target = (int(source[0] + dx), int(source[1] + dy))
            if target in occupied:
                continue
            old = best.get(target)
            if old is None or float(points["RCS"][index]) > float(points["RCS"][old]):
                best[target] = int(index)
    if not best:
        return np.zeros((0, 3), dtype=np.float32)
    rows = [((cell[0] + 0.5) * vx, (cell[1] + 0.5) * vy, float(z[index]))
            for cell, index in best.items()]
    return np.asarray(rows, dtype=np.float32)


def load_infos(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def main():
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    split_infos = [("train", item) for item in load_infos(args.raw_train)]
    split_infos += [("val", item) for item in load_infos(args.raw_val)]
    frames = []
    for split, info in split_infos:
        radar = info["radars"]["RADAR_FRONT"]
        pcd_path = Path(radar["radar_path"])
        if not pcd_path.is_absolute():
            pcd_path = source_root / pcd_path
        names = np.asarray([CLASS_MAP.get(str(name), str(name))
                            for name in info["annos"]["names"]])
        frames.append({
            "split": split,
            "frame_id": info.get("frame_id", ""),
            "points": read_pcd(pcd_path),
            "radar2body": np.asarray(radar["radar2body"], dtype=np.float32),
            "boxes": np.asarray(info["annos"]["boxes_3d"], dtype=np.float32),
            "names": names,
        })

    print("pattern min_abs_v min_rcs generated hit hit_rate Car LargeVehicle Cyclist")
    for pattern in args.patterns:
        offsets = OFFSET_PATTERNS[pattern]
        for min_abs_v in args.min_abs_v:
            for min_rcs in args.min_rcs:
                total = hits = 0
                class_hits = {name: 0 for name in ("Car", "LargeVehicle", "Cyclist")}
                cyclist_details = []
                for frame in frames:
                    generated = expand(
                        frame["points"], min_abs_v, min_rcs, args.max_range,
                        tuple(args.voxel_size), offsets,
                    )
                    body = transform_xyz(generated, frame["radar2body"])
                    inside = points_in_boxes(body, frame["boxes"])
                    any_inside = inside.any(axis=1) if inside.shape[1] else np.zeros(len(body), dtype=bool)
                    total += len(body)
                    hits += int(any_inside.sum())
                    for class_name in class_hits:
                        selected = np.flatnonzero(frame["names"] == class_name)
                        if len(selected):
                            class_hits[class_name] += int(inside[:, selected].any(axis=1).sum())
                    cyclist_boxes = np.flatnonzero(frame["names"] == "Cyclist")
                    if frame["split"] == "val" and len(cyclist_boxes):
                        cyclist_details.append((frame["frame_id"], [int(inside[:, i].sum()) for i in cyclist_boxes]))
                rate = hits / total if total else 0.0
                print(f"{pattern:6s} {min_abs_v:9.3f} {min_rcs:7.2f} {total:9d} "
                      f"{hits:6d} {rate:8.4f} {class_hits['Car']:5d} "
                      f"{class_hits['LargeVehicle']:12d} {class_hits['Cyclist']:7d}")
                if cyclist_details:
                    detail = "; ".join(f"{frame_id}:{counts}" for frame_id, counts in cyclist_details)
                    print(f"  val_cyclist {detail}")


if __name__ == "__main__":
    main()
