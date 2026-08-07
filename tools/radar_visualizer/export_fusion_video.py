#!/usr/bin/env python3
"""Export a rich HR-4D fusion visualization video without running Flask.

This uses the radar_visualizer data layer for synchronized camera, radar, ATX
LiDAR, EM4 LiDAR, GT boxes, and DET/TRACK overlays, then renders a dense
offline MP4 suitable for quick review.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = Path(__file__).resolve().with_name("server.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "1000_original_data")
    parser.add_argument("--pkl", type=Path, default=ROOT / "data" / "1000_original_data" / "splits" / "hr4d_1000_v1" / "infos_test_200.pkl")
    parser.add_argument("--overlay-json", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, default=ROOT / "output" / "weikang_tracking" / "fusion_visualization.mp4")
    parser.add_argument("--output-preview", type=Path, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--x-range", type=float, nargs=2, default=(-10.0, 220.0), metavar=("MIN", "MAX"))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-70.0, 70.0), metavar=("MIN", "MAX"))
    parser.add_argument("--point-radius", type=int, default=1)
    return parser.parse_args()


def load_visualizer_server():
    if "flask" not in sys.modules:
        flask = types.ModuleType("flask")
        flask.Flask = object
        flask.Response = object
        flask.abort = lambda *args, **kwargs: None
        flask.send_from_directory = lambda *args, **kwargs: None
        sys.modules["flask"] = flask

    spec = importlib.util.spec_from_file_location("hr4d_radar_visualizer_server", SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {SERVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_from_bytes(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode visualizer image bytes")
    return image


def resize_cover(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    ratio = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * ratio), round(image.shape[0] * ratio)), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), (18, 20, 23), dtype=np.uint8)
    x0 = (width - resized.shape[1]) // 2
    y0 = (height - resized.shape[0]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas


def draw_segments(image: np.ndarray, segments: list, color: tuple[int, int, int], thickness: int = 2) -> None:
    for segment in segments:
        pts = np.asarray(segment, dtype=np.float32)
        if pts.shape != (2, 2):
            continue
        p0 = tuple(np.round(pts[0]).astype(int))
        p1 = tuple(np.round(pts[1]).astype(int))
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


class BevPanel:
    def __init__(self, rect: tuple[int, int, int, int], x_range: tuple[float, float], y_range: tuple[float, float]):
        self.x, self.y, self.w, self.h = rect
        self.x_range = tuple(float(v) for v in x_range)
        self.y_range = tuple(float(v) for v in y_range)
        self.margin_l = 46
        self.margin_t = 34
        self.margin_r = 18
        self.margin_b = 24
        plot_w = self.w - self.margin_l - self.margin_r
        plot_h = self.h - self.margin_t - self.margin_b
        self.scale = min(plot_w / (self.y_range[1] - self.y_range[0]), plot_h / (self.x_range[1] - self.x_range[0]))
        self.pad_x = (plot_w - (self.y_range[1] - self.y_range[0]) * self.scale) * 0.5
        self.pad_y = (plot_h - (self.x_range[1] - self.x_range[0]) * self.scale) * 0.5

    def world_to_px(self, x: float, y: float) -> tuple[int, int]:
        u = self.x + self.margin_l + self.pad_x + (self.y_range[1] - y) * self.scale
        v = self.y + self.margin_t + self.pad_y + (self.x_range[1] - x) * self.scale
        return int(round(u)), int(round(v))

    def draw_base(self, canvas: np.ndarray, title: str) -> None:
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), (27, 30, 35), -1)
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), (58, 64, 74), 1)
        cv2.putText(canvas, title, (self.x + 14, self.y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 242, 245), 1, cv2.LINE_AA)
        for x in np.arange(np.ceil(self.x_range[0] / 20.0) * 20.0, self.x_range[1] + 1, 20.0):
            cv2.line(canvas, self.world_to_px(x, self.y_range[0]), self.world_to_px(x, self.y_range[1]), (48, 53, 61), 1)
            cv2.putText(canvas, f"{x:.0f}", self.world_to_px(x, self.y_range[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (124, 130, 140), 1)
        for y in np.arange(np.ceil(self.y_range[0] / 20.0) * 20.0, self.y_range[1] + 1, 20.0):
            cv2.line(canvas, self.world_to_px(self.x_range[0], y), self.world_to_px(self.x_range[1], y), (48, 53, 61), 1)
        cv2.line(canvas, self.world_to_px(self.x_range[0], 0), self.world_to_px(self.x_range[1], 0), (80, 88, 100), 1)
        ego = np.array([
            self.world_to_px(3, 0),
            self.world_to_px(-2, 2.2),
            self.world_to_px(-2, -2.2),
        ], dtype=np.int32)
        cv2.fillConvexPoly(canvas, ego, (230, 230, 230))

    def draw_points(self, canvas: np.ndarray, points: dict, color: tuple[int, int, int], radius: int, values: list | None = None) -> int:
        xs = np.asarray(points.get("x", []), dtype=np.float32)
        ys = np.asarray(points.get("y", []), dtype=np.float32)
        if len(xs) == 0:
            return 0
        mask = (
            np.isfinite(xs) & np.isfinite(ys)
            & (xs >= self.x_range[0]) & (xs <= self.x_range[1])
            & (ys >= self.y_range[0]) & (ys <= self.y_range[1])
        )
        xs = xs[mask]
        ys = ys[mask]
        if values is not None:
            values_arr = np.asarray(values, dtype=np.float32)[mask]
            colors = colorize(values_arr)
        else:
            colors = np.tile(np.asarray(color, dtype=np.uint8), (len(xs), 1))
        for x, y, point_color in zip(xs, ys, colors):
            cv2.circle(canvas, self.world_to_px(float(x), float(y)), radius, tuple(int(c) for c in point_color), -1, cv2.LINE_AA)
        return len(xs)

    def draw_box(self, canvas: np.ndarray, box: list, color: tuple[int, int, int], thickness: int = 2) -> None:
        corners = corners_bev(np.asarray(box, dtype=np.float32))
        pixels = np.asarray([self.world_to_px(float(x), float(y)) for x, y in corners], dtype=np.int32)
        cv2.polylines(canvas, [pixels], True, color, thickness, cv2.LINE_AA)
        x, y, _, length, _, _, yaw = [float(v) for v in box[:7]]
        front = (x + np.cos(yaw) * length * 0.5, y + np.sin(yaw) * length * 0.5)
        cv2.line(canvas, self.world_to_px(x, y), self.world_to_px(*front), color, thickness, cv2.LINE_AA)


def colorize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    finite = np.isfinite(values)
    if not finite.any():
        normalized = np.zeros_like(values, dtype=np.uint8)
    else:
        lo, hi = np.nanpercentile(values[finite], [5, 95])
        normalized = np.clip((values - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)


def corners_bev(box: np.ndarray) -> np.ndarray:
    x, y, _, length, width, _, yaw = [float(v) for v in box[:7]]
    local = np.array([[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]], dtype=np.float32)
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def render_frame(dataset, index: int, args: argparse.Namespace) -> np.ndarray:
    frame = dataset.frame(index)
    canvas = np.full((args.height, args.width, 3), (18, 20, 23), dtype=np.uint8)

    camera_w, camera_h = 1280, 720
    camera = resize_cover(image_from_bytes(dataset.image_bytes(index)), (camera_w, camera_h))
    radar_projection = frame.get("radar_projection", {})
    for u, v, depth in zip(radar_projection.get("u", []), radar_projection.get("v", []), radar_projection.get("depth", [])):
        if depth < 180:
            cv2.circle(camera, (int(u), int(v)), 1, (255, 220, 60), -1)
    for anno in frame.get("annotations", []):
        draw_segments(camera, anno.get("image_segments", []), (80, 210, 120), 2)
    for overlay in frame.get("overlays", []):
        is_track = bool(overlay.get("track_id")) or overlay.get("source") == "track"
        draw_segments(camera, overlay.get("image_segments", []), (40, 230, 255) if is_track else (255, 80, 190), 2)
    canvas[0:camera_h, 0:camera_w] = camera

    fused = BevPanel((1280, 0, 640, 720), tuple(args.x_range), tuple(args.y_range))
    fused.draw_base(canvas, "Fused BEV: Radar + ATX + EM4 + GT + DET/TRACK")
    counts = draw_sensor_points(canvas, fused, frame, args.point_radius)
    draw_boxes(canvas, fused, frame, include_gt=True, include_overlays=True)

    panel_w = args.width // 3
    panels = [
        ("Radar FRONT doppler/RCS", "radar_front", (0, 720, panel_w, 360), "doppler"),
        ("ATX LiDAR FRONT_2", "lidar_front_2", (panel_w, 720, panel_w, 360), None),
        ("EM4 LiDAR FRONT", "lidar_front", (panel_w * 2, 720, args.width - panel_w * 2, 360), None),
    ]
    for title, sensor_name, rect, feature_name in panels:
        panel = BevPanel(rect, tuple(args.x_range), tuple(args.y_range))
        panel.draw_base(canvas, title)
        sensor = frame["sensors"].get(sensor_name, {})
        values = sensor.get("features", {}).get(feature_name) if feature_name else None
        count = panel.draw_points(canvas, sensor.get("points", {}), sensor_color(sensor_name), args.point_radius, values)
        cv2.putText(canvas, f"visible {count} / source {sensor.get('source_count', 0)}", (rect[0] + 14, rect[1] + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (178, 184, 194), 1, cv2.LINE_AA)
        draw_boxes(canvas, panel, frame, include_gt=False, include_overlays=True)

    draw_header(canvas, frame, counts)
    return canvas


def draw_sensor_points(canvas: np.ndarray, panel: BevPanel, frame: dict, radius: int) -> dict[str, int]:
    counts = {}
    for sensor_name in ("lidar_front_2", "lidar_front", "radar_front"):
        sensor = frame["sensors"].get(sensor_name, {})
        values = sensor.get("features", {}).get("doppler") if sensor_name == "radar_front" else None
        counts[sensor_name] = panel.draw_points(canvas, sensor.get("points", {}), sensor_color(sensor_name), radius, values)
    return counts


def sensor_color(sensor_name: str) -> tuple[int, int, int]:
    return {
        "radar_front": (255, 220, 60),
        "lidar_front_2": (90, 170, 255),
        "lidar_front": (205, 130, 255),
    }.get(sensor_name, (180, 180, 180))


def draw_boxes(canvas: np.ndarray, panel: BevPanel, frame: dict, include_gt: bool, include_overlays: bool) -> None:
    if include_gt:
        for anno in frame.get("annotations", []):
            panel.draw_box(canvas, anno["box"], (80, 210, 120), 1)
    if include_overlays:
        for overlay in frame.get("overlays", []):
            is_track = bool(overlay.get("track_id")) or overlay.get("source") == "track"
            color = (40, 230, 255) if is_track else (255, 80, 190)
            panel.draw_box(canvas, overlay["box"], color, 2 if is_track else 1)


def draw_header(canvas: np.ndarray, frame: dict, counts: dict[str, int]) -> None:
    cv2.rectangle(canvas, (0, 0), (1280, 72), (18, 20, 23), -1)
    title = f"HR-4D Fusion Visualization | {frame['sequence_id']} | frame {frame['frame_id'][:8]}"
    cv2.putText(canvas, title, (24, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 250), 1, cv2.LINE_AA)
    overlays = frame.get("overlays", [])
    tracks = sum(1 for item in overlays if item.get("track_id") or item.get("source") == "track")
    dets = len(overlays) - tracks
    sub = (
        f"Radar {counts.get('radar_front', 0)} | ATX {counts.get('lidar_front_2', 0)} | "
        f"EM4 {counts.get('lidar_front', 0)} | GT {len(frame.get('annotations', []))} | DET {dets} | TRACK {tracks}"
    )
    cv2.putText(canvas, sub, (24, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (188, 194, 204), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Radar projection points are shown on the camera image.", (820, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (188, 194, 204), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    server = load_visualizer_server()
    dataset = server.Dataset(args.data_root, args.pkl, args.overlay_json)
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    frame_count = len(dataset.infos) - args.start_index
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer: {args.output_video}")
    preview = None
    for offset in range(frame_count):
        index = args.start_index + offset
        image = render_frame(dataset, index, args)
        writer.write(image)
        if preview is None and len(dataset.frame(index).get("overlays", [])):
            preview = image.copy()
    writer.release()
    if args.output_preview:
        args.output_preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output_preview), preview if preview is not None else render_frame(dataset, args.start_index, args))
    print(f"frames={frame_count}")
    print(f"wrote_video={args.output_video}")
    if args.output_preview:
        print(f"wrote_preview={args.output_preview}")


if __name__ == "__main__":
    main()
