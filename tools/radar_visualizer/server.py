#!/usr/bin/env python3
"""Local web visualizer for the provided 4D radar PKL dataset."""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, abort, send_from_directory
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = APP_DIR.parents[1] / "data" / "1000_original_data"
CAMERA_KEY = "CAMERA_FRONT_WIDE"
CAMERA_LABEL = "前广图像 / CAMERA_FRONT_WIDE"
CALIBRATION_IMAGE_SIZE = (1920, 1080)
DISPLAY_IMAGE_SIZE = (1280, 720)
OUTPUT_COORDINATE_FRAME = "body_flu@camera_front_wide_timestamp"
SENSORS = {
    "radar_front": ("radars", "RADAR_FRONT", "radar_path", "radar2body", 20_000),
    "lidar_front_2": ("lidars", "LIDAR_FRONT_2", "lidar_path", "lidar2body", 30_000),
    "lidar_front": ("lidars", "LIDAR_FRONT", "lidar_path", "lidar2body", 30_000),
}
SENSOR_LABELS = {
    "radar_front": "4D Radar / radar_front",
    "lidar_front_2": "ATX / lidar_front_2",
    "lidar_front": "EM4 / lidar_front",
}
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
TIMESTAMP_PATTERN = re.compile(r"__(\d{13})_")
PCD_TYPES = {
    "F": {4: "<f4", 8: "<f8"},
    "U": {1: "u1", 2: "<u2", 4: "<u4", 8: "<u8"},
    "I": {1: "i1", 2: "<i2", 4: "<i4", 8: "<i8"},
}


def json_response(payload: Any, status: int = 200) -> Response:
    return Response(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        status=status,
        mimetype="application/json",
    )


def clean_number(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def array_list(values: np.ndarray, decimals: int = 4) -> list:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.floating):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.round(values, decimals)
    return values.tolist()


def timestamp_from_path(path: str, fallback: float) -> float:
    match = TIMESTAMP_PATTERN.search(path)
    return int(match.group(1)) / 1000.0 if match else float(fallback)


def axis_angle_rotation(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = rotvec / angle
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def interpolate_pose(samples: list[tuple[float, np.ndarray]], timestamp: float) -> np.ndarray:
    unique = {}
    for sample_time, pose in samples:
        unique[float(sample_time)] = np.asarray(pose, dtype=np.float64)
    ordered = sorted(unique.items(), key=lambda item: item[0])
    if not ordered:
        raise ValueError("No IMU pose samples available for synchronization")
    if len(ordered) == 1:
        return ordered[0][1].copy()

    times = np.asarray([item[0] for item in ordered])
    upper = int(np.searchsorted(times, timestamp, side="right"))
    if upper == 0:
        lower, upper = 0, 1
    elif upper >= len(ordered):
        lower, upper = len(ordered) - 2, len(ordered) - 1
    else:
        lower = upper - 1

    time0, pose0 = ordered[lower]
    time1, pose1 = ordered[upper]
    alpha = (timestamp - time0) / max(time1 - time0, 1e-9)
    relative = pose0[:3, :3].T @ pose1[:3, :3]
    cos_angle = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cos_angle)
    if angle < 1e-10:
        rotvec = np.zeros(3)
    else:
        axis = np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        ) / (2.0 * math.sin(angle))
        rotvec = axis * angle

    pose = np.eye(4)
    pose[:3, :3] = pose0[:3, :3] @ axis_angle_rotation(rotvec * alpha)
    pose[:3, 3] = pose0[:3, 3] + alpha * (pose1[:3, 3] - pose0[:3, 3])
    return pose


def body_time_transform(source_pose: np.ndarray, target_pose: np.ndarray, imu2body: np.ndarray) -> np.ndarray:
    return imu2body @ np.linalg.inv(target_pose) @ source_pose @ np.linalg.inv(imu2body)


def synchronize_body_points(
    points: np.ndarray,
    timestamps: np.ndarray | float,
    target_pose: np.ndarray,
    pose_samples: list[tuple[float, np.ndarray]],
    imu2body: np.ndarray,
) -> np.ndarray:
    if not len(points):
        return points
    timestamps = np.broadcast_to(np.asarray(timestamps, dtype=np.float64), (len(points),))
    synchronized = np.empty_like(points, dtype=np.float64)
    unique_times, inverse = np.unique(timestamps, return_inverse=True)
    for time_index, source_time in enumerate(unique_times):
        transform = body_time_transform(interpolate_pose(pose_samples, float(source_time)), target_pose, imu2body)
        mask = inverse == time_index
        homogeneous = np.column_stack((points[mask, :3], np.ones(np.count_nonzero(mask))))
        synchronized[mask] = (homogeneous @ transform.T)[:, :3]
    return synchronized


def synchronize_sensor_vectors(
    vectors: np.ndarray,
    sensor2body: np.ndarray,
    timestamp: float,
    target_pose: np.ndarray,
    pose_samples: list[tuple[float, np.ndarray]],
    imu2body: np.ndarray,
) -> np.ndarray:
    source_body_vectors = np.asarray(vectors, dtype=np.float64) @ sensor2body[:3, :3].T
    time_rotation = body_time_transform(
        interpolate_pose(pose_samples, timestamp),
        target_pose,
        imu2body,
    )[:3, :3]
    return source_body_vectors @ time_rotation.T


def box_corners(box: np.ndarray) -> np.ndarray:
    x, y, z, length, width, height, yaw = [float(v) for v in box]
    local = np.array(
        [
            [length / 2, width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [length / 2, width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [-length / 2, -width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return local @ rotation.T + np.array([x, y, z])


def project_points(points: np.ndarray, camera: dict, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if not len(points):
        return np.empty((0, 2)), np.empty((0,))
    body_to_camera = np.linalg.inv(np.asarray(camera["camera2body"], dtype=np.float64))
    homogeneous = np.column_stack((points[:, :3], np.ones(len(points))))
    camera_points = homogeneous @ body_to_camera.T
    depth = camera_points[:, 2]
    projected = camera_points[:, :3] @ np.asarray(camera["camera_intrinsic"], dtype=np.float64).T
    uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)
    uv *= np.asarray(image_size, dtype=np.float64) / np.asarray(CALIBRATION_IMAGE_SIZE, dtype=np.float64)
    width, height = image_size
    visible = (
        (depth > 0.1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv[visible], visible


def parse_pcd(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header: dict[str, list[str]] = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Invalid PCD header: {path}")
            parts = line.decode("utf-8", errors="replace").strip().split()
            if parts and not parts[0].startswith("#"):
                header[parts[0]] = parts[1:]
            if line.startswith(b"DATA "):
                break

        data_mode = header["DATA"][0]
        if data_mode != "binary":
            raise ValueError(f"Only binary PCD is supported, got {data_mode}: {path}")

        fields = header["FIELDS"]
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        dtype_fields = []
        for name, size, type_name, count in zip(fields, sizes, types, counts):
            dtype = PCD_TYPES[type_name][size]
            dtype_fields.append((name, dtype, (count,)) if count > 1 else (name, dtype))
        return np.fromfile(stream, dtype=np.dtype(dtype_fields), count=int(header["POINTS"][0]))


class Dataset:
    def __init__(self, data_root: Path, pkl_path: Path, overlay_path: Path | None = None):
        self.data_root = data_root.resolve()
        self.pkl_path = pkl_path.resolve()
        with self.pkl_path.open("rb") as stream:
            self.infos: list[dict] = pickle.load(stream)
        self.overlay_path = overlay_path.resolve() if overlay_path else None
        self.overlays = self._load_overlays(self.overlay_path)
        self.frames = [self._frame_summary(index, info) for index, info in enumerate(self.infos)]
        self.available_frames = [frame for frame in self.frames if frame["available"]["camera_front_wide"]]

    def local_path(self, relative_path: str) -> Path:
        path = (self.data_root / relative_path).resolve()
        if self.data_root not in path.parents and path != self.data_root:
            raise ValueError("Path escapes data root")
        return path

    def _frame_summary(self, index: int, info: dict) -> dict:
        available = {}
        for sensor_name, (group, key, path_key, _, _) in SENSORS.items():
            available[sensor_name] = self.local_path(info[group][key][path_key]).is_file()
        image_path = info["cameras"][CAMERA_KEY]["image_path"]
        available["camera_front_wide"] = self.local_path(image_path).is_file()
        return {
            "index": index,
            "sequence_id": info["sequence_id"],
            "frame_id": info["frame_id"],
            "timestamp": clean_number(info["timestamp"]),
            "boxes": len(info["annos"]["names"]),
            "available": available,
            "complete": all(available.values()),
        }

    def meta(self) -> dict:
        classes: dict[str, int] = {}
        for info in self.infos:
            for name in info["annos"]["names"]:
                classes[str(name)] = classes.get(str(name), 0) + 1
        return {
            "pkl_path": str(self.pkl_path),
            "data_root": str(self.data_root),
            "total_pkl_frames": len(self.infos),
            "available_frame_count": len(self.available_frames),
            "complete_frame_count": sum(frame["complete"] for frame in self.frames),
            "frames": self.available_frames,
            "classes": classes,
            "sensors": SENSOR_LABELS,
            "camera": CAMERA_LABEL,
            "display_image_size": DISPLAY_IMAGE_SIZE,
            "overlay": {
                "enabled": self.overlay_path is not None,
                "path": str(self.overlay_path) if self.overlay_path else None,
                "frame_count": len(self.overlays),
            },
        }

    @lru_cache(maxsize=32)
    def image_bytes(self, index: int) -> bytes:
        if index < 0 or index >= len(self.infos):
            raise IndexError(index)
        path = self.local_path(self.infos[index]["cameras"][CAMERA_KEY]["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail(DISPLAY_IMAGE_SIZE, Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=82)
            return output.getvalue()

    @lru_cache(maxsize=12)
    def frame(self, index: int) -> dict:
        if index < 0 or index >= len(self.infos):
            raise IndexError(index)
        info = self.infos[index]
        camera = info["cameras"][CAMERA_KEY]
        camera_timestamp = timestamp_from_path(camera["image_path"], info["timestamp"])
        imu2body = np.asarray(info["imu"]["imu2body"], dtype=np.float64)
        pose_samples = [
            (float(info["timestamp"]), np.asarray(info["imu"]["imu2world"], dtype=np.float64)),
            *[
                (float(radar["timestamp"]), np.asarray(radar["imu2world"], dtype=np.float64))
                for radar in info["radars"].values()
            ],
        ]
        camera_pose = interpolate_pose(pose_samples, camera_timestamp)
        image_path = self.local_path(camera["image_path"])
        image_size = (0, 0)
        if image_path.is_file():
            with Image.open(image_path) as image:
                ratio = min(DISPLAY_IMAGE_SIZE[0] / image.width, DISPLAY_IMAGE_SIZE[1] / image.height, 1.0)
                image_size = (round(image.width * ratio), round(image.height * ratio))

        sensor_payload = {}
        radar_body_points = np.empty((0, 3), dtype=np.float32)
        radar_features: dict[str, np.ndarray] = {}
        for sensor_name, config in SENSORS.items():
            group, key, path_key, transform_key, limit = config
            sensor = info[group][key]
            path = self.local_path(sensor[path_key])
            if not path.is_file():
                sensor_payload[sensor_name] = {"available": False, "label": SENSOR_LABELS[sensor_name]}
                continue
            cloud = parse_pcd(path)
            source_count = len(cloud)
            cloud = self._downsample(cloud, limit)
            xyz = np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(np.float64)
            valid = np.isfinite(xyz).all(axis=1)
            cloud = cloud[valid]
            xyz = xyz[valid]
            transform = np.asarray(sensor[transform_key], dtype=np.float64)
            body = np.column_stack((xyz, np.ones(len(xyz)))) @ transform.T
            body = body[:, :3]
            if sensor_name == "radar_front":
                source_timestamps = float(sensor["timestamp"])
                sync_mode = "frame_pose"
            elif "timestamp" in cloud.dtype.names:
                source_timestamps = np.asarray(cloud["timestamp"], dtype=np.float64)
                sync_mode = "per_point" if len(np.unique(source_timestamps)) > 1 else "frame_pose"
            else:
                source_timestamps = timestamp_from_path(sensor[path_key], info["timestamp"])
                sync_mode = "frame_pose"
            body = synchronize_body_points(body, source_timestamps, camera_pose, pose_samples, imu2body).astype(np.float32)
            features = self._features(cloud, sensor_name)
            if sensor_name == "radar_front" and {"Vx", "Vy"} <= features.keys():
                body_velocity = synchronize_sensor_vectors(
                    np.column_stack((features["Vx"], features["Vy"], np.zeros(len(body)))),
                    transform,
                    float(sensor["timestamp"]),
                    camera_pose,
                    pose_samples,
                    imu2body,
                )
                features["Vx_sensor"] = features["Vx"]
                features["Vy_sensor"] = features["Vy"]
                features["Vx"] = body_velocity[:, 0].astype(np.float32)
                features["Vy"] = body_velocity[:, 1].astype(np.float32)
            source_timestamps = np.broadcast_to(np.asarray(source_timestamps, dtype=np.float64), (len(body),))
            sensor_payload[sensor_name] = {
                "available": True,
                "label": SENSOR_LABELS[sensor_name],
                "coordinate_frame": OUTPUT_COORDINATE_FRAME,
                "source_count": int(source_count),
                "display_count": int(len(body)),
                "path": sensor[path_key],
                "points": {
                    "x": array_list(body[:, 0]),
                    "y": array_list(body[:, 1]),
                    "z": array_list(body[:, 2]),
                },
                "features": {name: array_list(values) for name, values in features.items()},
                "feature_stats": self._feature_stats(features),
                "synchronization": {
                    "target": "camera_front_wide",
                    "mode": sync_mode,
                    "offset_ms": {
                        "min": clean_number((np.min(source_timestamps) - camera_timestamp) * 1000.0),
                        "median": clean_number((np.median(source_timestamps) - camera_timestamp) * 1000.0),
                        "max": clean_number((np.max(source_timestamps) - camera_timestamp) * 1000.0),
                    },
                },
            }
            if sensor_name == "radar_front":
                radar_body_points = body
                radar_features = features

        radar_projection = {"u": [], "v": [], "depth": [], "features": {}}
        if image_size != (0, 0) and len(radar_body_points):
            uv, visible = project_points(radar_body_points, camera, image_size)
            camera_origin = np.asarray(camera["camera2body"], dtype=np.float64)[:3, 3]
            depth = np.linalg.norm(radar_body_points[visible] - camera_origin, axis=1)
            radar_projection = {
                "u": array_list(uv[:, 0], 2),
                "v": array_list(uv[:, 1], 2),
                "depth": array_list(depth, 2),
                "features": {name: array_list(values[visible]) for name, values in radar_features.items()},
            }

        annotation_transform = body_time_transform(
            np.asarray(info["imu"]["imu2world"], dtype=np.float64),
            camera_pose,
            imu2body,
        )
        annotations = self._annotations(info, camera, image_size, annotation_transform)
        overlays = self._overlays_for_frame(index, info, camera, image_size, annotation_transform)
        image_url = f"/api/image/{index}" if image_path.is_file() else None
        return {
            "index": index,
            "sequence_id": info["sequence_id"],
            "frame_id": info["frame_id"],
            "timestamp": clean_number(info["timestamp"]),
            "camera_timestamp": clean_number(camera_timestamp),
            "tags": [str(tag) for tag in info.get("tags", [])],
            "image": {
                "available": image_path.is_file(),
                "url": image_url,
                "path": camera["image_path"],
                "width": image_size[0],
                "height": image_size[1],
                "label": CAMERA_LABEL,
            },
            "sensors": sensor_payload,
            "radar_projection": radar_projection,
            "annotations": annotations,
            "overlays": overlays,
            "coordinate_frame": OUTPUT_COORDINATE_FRAME,
            "synchronization": {
                "target": CAMERA_LABEL,
                "camera_timestamp": clean_number(camera_timestamp),
                "annotation_offset_ms": clean_number((float(info["timestamp"]) - camera_timestamp) * 1000.0),
            },
        }

    @staticmethod
    def _load_overlays(overlay_path: Path | None) -> dict[str, list[dict]]:
        if overlay_path is None:
            return {}
        with overlay_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict) and isinstance(payload.get("frames"), dict):
            return {str(key): list(value or []) for key, value in payload["frames"].items()}
        if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
            result: dict[str, list[dict]] = {}
            for frame in payload["frames"]:
                frame_key = frame.get("frame_id", frame.get("index"))
                if frame_key is None:
                    continue
                objects = list(frame.get("objects", []) or [])
                detections = [dict(item, source=item.get("source", "det")) for item in frame.get("detections", []) or []]
                tracks = [dict(item, source=item.get("source", "track")) for item in frame.get("tracks", []) or []]
                result[str(frame_key)] = [*objects, *detections, *tracks]
            return result
        if isinstance(payload, list):
            result: dict[str, list[dict]] = {}
            for item in payload:
                frame_key = item.get("frame_id", item.get("index"))
                if frame_key is None:
                    continue
                result.setdefault(str(frame_key), []).append(item)
            return result
        raise ValueError(f"Unsupported overlay JSON format: {overlay_path}")

    def _overlays_for_frame(
        self,
        index: int,
        info: dict,
        camera: dict,
        image_size: tuple[int, int],
        time_transform: np.ndarray,
    ) -> list[dict]:
        if not self.overlays:
            return []
        raw_items = [
            *self.overlays.get(str(info.get("frame_id")), []),
            *self.overlays.get(str(info.get("sequence_id")), []),
        ]
        index_items = self.overlays.get(str(index), [])
        if index_items:
            raw_items.extend(index_items)

        result = []
        for item_index, item in enumerate(raw_items):
            box = self._overlay_box(item)
            if box is None:
                continue
            synchronized_box = np.asarray(box, dtype=np.float64).copy()
            center = time_transform @ np.array([*synchronized_box[:3], 1.0])
            heading = time_transform[:3, :3] @ np.array(
                [math.cos(synchronized_box[6]), math.sin(synchronized_box[6]), 0.0]
            )
            synchronized_box[:3] = center[:3]
            synchronized_box[6] = math.atan2(heading[1], heading[0])
            corners = box_corners(synchronized_box)
            result.append(
                {
                    "name": str(item.get("name", item.get("class_name", item.get("label", "det")))),
                    "track_id": str(item.get("track_id", item.get("id", ""))),
                    "score": clean_number(item.get("score", 1.0)),
                    "source": str(item.get("source", item.get("type", "det"))).lower(),
                    "eval_id": str(item.get("eval_id", "")),
                    "match_status": str(item.get("match_status", "")),
                    "reason": str(item.get("reason", "")),
                    "pred_index": item.get("pred_index"),
                    "gt_index": item.get("gt_index"),
                    "class_name": str(item.get("class_name", item.get("name", ""))),
                    "box": array_list(synchronized_box, 4),
                    "corners": array_list(corners, 4),
                    "image_segments": self._box_image_segments(corners, camera, image_size),
                    "raw_index": item_index,
                }
            )
        return result

    @staticmethod
    def _overlay_box(item: dict) -> np.ndarray | None:
        for key in ("box", "box3d", "boxes_3d", "boxes_lidar"):
            if key in item:
                values = np.asarray(item[key], dtype=np.float64).reshape(-1)
                return values[:7] if len(values) >= 7 else None
        return None

    @staticmethod
    def _box_image_segments(corners: np.ndarray, camera: dict, image_size: tuple[int, int]) -> list:
        if image_size == (0, 0):
            return []
        body_to_camera = np.linalg.inv(np.asarray(camera["camera2body"], dtype=np.float64))
        intrinsic = np.asarray(camera["camera_intrinsic"], dtype=np.float64)
        camera_points = np.column_stack((corners, np.ones(8))) @ body_to_camera.T
        projected = camera_points[:, :3] @ intrinsic.T
        uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)
        uv *= np.asarray(image_size, dtype=np.float64) / np.asarray(CALIBRATION_IMAGE_SIZE, dtype=np.float64)
        segments = []
        for start, end in BOX_EDGES:
            if camera_points[start, 2] > 0.1 and camera_points[end, 2] > 0.1:
                segments.append(array_list(np.array([uv[start], uv[end]]), 2))
        return segments

    @staticmethod
    def _downsample(cloud: np.ndarray, limit: int) -> np.ndarray:
        if len(cloud) <= limit:
            return cloud
        indices = np.linspace(0, len(cloud) - 1, limit, dtype=np.int64)
        return cloud[indices]

    @staticmethod
    def _features(cloud: np.ndarray, sensor_name: str) -> dict[str, np.ndarray]:
        wanted = (
            ("doppler", "RCS", "power", "AbsV", "Vx", "Vy", "range")
            if sensor_name == "radar_front"
            else ("intensity", "ring")
        )
        return {
            field: np.nan_to_num(np.asarray(cloud[field], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            for field in wanted
            if field in cloud.dtype.names
        }

    @staticmethod
    def _feature_stats(features: dict[str, np.ndarray]) -> dict:
        result = {}
        for name, values in features.items():
            if len(values):
                result[name] = {
                    "min": clean_number(np.min(values)),
                    "median": clean_number(np.median(values)),
                    "max": clean_number(np.max(values)),
                }
        return result

    @staticmethod
    def _annotations(
        info: dict,
        camera: dict,
        image_size: tuple[int, int],
        time_transform: np.ndarray,
    ) -> list[dict]:
        annos = info["annos"]
        num_pts = annos.get("num_pts", {})
        body_to_camera = np.linalg.inv(np.asarray(camera["camera2body"], dtype=np.float64))
        intrinsic = np.asarray(camera["camera_intrinsic"], dtype=np.float64)
        result = []
        for index, (name, box, gt_id) in enumerate(zip(annos["names"], annos["boxes_3d"], annos["gt_id"])):
            synchronized_box = np.asarray(box, dtype=np.float64).copy()
            center = time_transform @ np.array([*synchronized_box[:3], 1.0])
            heading = time_transform[:3, :3] @ np.array(
                [math.cos(synchronized_box[6]), math.sin(synchronized_box[6]), 0.0]
            )
            synchronized_box[:3] = center[:3]
            synchronized_box[6] = math.atan2(heading[1], heading[0])
            corners = box_corners(synchronized_box)
            segments = []
            if image_size != (0, 0):
                camera_points = np.column_stack((corners, np.ones(8))) @ body_to_camera.T
                projected = camera_points[:, :3] @ intrinsic.T
                uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)
                uv *= np.asarray(image_size, dtype=np.float64) / np.asarray(CALIBRATION_IMAGE_SIZE, dtype=np.float64)
                for start, end in BOX_EDGES:
                    if camera_points[start, 2] > 0.1 and camera_points[end, 2] > 0.1:
                        segments.append(array_list(np.array([uv[start], uv[end]]), 2))
            result.append(
                {
                    "name": str(name),
                    "id": str(gt_id),
                    "box": array_list(synchronized_box, 4),
                    "corners": array_list(corners, 4),
                    "image_segments": segments,
                    "num_pts": {
                        key: int(np.asarray(values)[index])
                        for key, values in num_pts.items()
                        if index < len(values)
                    },
                }
            )
        return result


def create_app(dataset: Dataset) -> Flask:
    app = Flask(__name__, static_folder=str(APP_DIR / "static"), static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/meta")
    def meta():
        return json_response(dataset.meta())

    @app.get("/api/frame/<int:index>")
    def frame(index: int):
        try:
            return json_response(dataset.frame(index))
        except IndexError:
            abort(404)
        except Exception as exc:
            return json_response({"error": str(exc), "index": index}, 500)

    @app.get("/api/image/<int:index>")
    def image(index: int):
        if index < 0 or index >= len(dataset.infos):
            abort(404)
        path = dataset.local_path(dataset.infos[index]["cameras"][CAMERA_KEY]["image_path"])
        if not path.is_file():
            abort(404)
        return Response(
            dataset.image_bytes(index),
            mimetype="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pkl", type=Path, default=None, help="Defaults to <data-root>/infos_test_1000.pkl")
    parser.add_argument("--overlay-json", type=Path, default=None, help="Optional DET/TRACK overlay JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pkl_path = args.pkl or args.data_root / "infos_test_1000.pkl"
    dataset = Dataset(args.data_root, pkl_path, args.overlay_json)
    print(
        f"Loaded {len(dataset.infos)} PKL frames; "
        f"{len(dataset.available_frames)} have local data; "
        f"{sum(frame['complete'] for frame in dataset.frames)} are complete."
    )
    print(f"Open http://{args.host}:{args.port}")
    create_app(dataset).run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
