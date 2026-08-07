#!/usr/bin/env python3
"""Offline DET -> TRACK post-processing and BEV video export for HR-4D.

The script intentionally stays lightweight: it consumes OpenPCDet-style
prediction PKLs, applies a constant-velocity BEV tracker, and writes track
overlays plus an optional MP4. It can also use GT boxes as detections for
smoke testing the visualization/tracking chain before a model result is ready.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy is available in the HR-4D image.
    linear_sum_assignment = None


DEFAULT_INFOS = "data/1000_original_data/splits/hr4d_1000_v1/infos_test_200.pkl"
DEFAULT_OUTPUT_DIR = "output/weikang_tracking"
DEFAULT_CLASSES = ("Vehicle", "Pedestrian", "Cyclist", "Truck", "Bus", "Cone")
COLORS = {
    "detection": (70, 130, 255),
    "track": (55, 225, 255),
    "inactive": (120, 120, 120),
    "grid": (54, 58, 66),
    "axis": (210, 210, 210),
    "ego": (235, 235, 235),
    "text": (245, 245, 245),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight offline tracking on HR-4D detections and render a BEV MP4."
    )
    parser.add_argument("--infos", default=DEFAULT_INFOS, help="HR-4D infos PKL used for frame order.")
    parser.add_argument(
        "--predictions",
        default=None,
        help="OpenPCDet result.pkl. If omitted, use --use-gt-as-detections for smoke tests.",
    )
    parser.add_argument(
        "--use-gt-as-detections",
        action="store_true",
        help="Use GT boxes as detections. This is only for chain smoke/exploratory checks.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for JSON/PKL/MP4 outputs.")
    parser.add_argument("--output-json", default=None, help="Track overlay JSON path.")
    parser.add_argument("--output-pkl", default=None, help="Tracked OpenPCDet-style PKL path.")
    parser.add_argument("--output-video", default=None, help="BEV MP4 path. Omit to skip video.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames for quick smoke runs; <=0 uses all.")
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--distance-threshold", type=float, default=6.0, help="Max BEV association distance in meters.")
    parser.add_argument("--max-age", type=int, default=3, help="Frames to keep unmatched tracks alive.")
    parser.add_argument(
        "--max-time-gap",
        type=float,
        default=2.0,
        help="Reset tracker when frame timestamp gap exceeds this many seconds; <=0 disables gap reset.",
    )
    parser.add_argument("--min-hits", type=int, default=1, help="Minimum hits before a track is emitted.")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--bev-width", type=int, default=900)
    parser.add_argument("--bev-height", type=int, default=1400)
    parser.add_argument("--x-range", type=float, nargs=2, default=(-10.0, 220.0), metavar=("MIN", "MAX"))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-70.0, 70.0), metavar=("MIN", "MAX"))
    parser.add_argument("--draw-detections", action="store_true", help="Draw raw detections under tracks.")
    return parser.parse_args()


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def dump_pickle(path: str | Path, payload: Any) -> None:
    with Path(path).open("wb") as stream:
        pickle.dump(payload, stream)


def as_array(values: Any, width: int | None = None) -> np.ndarray:
    if values is None:
        return np.zeros((0, width or 0), dtype=np.float32)
    array = np.asarray(values)
    if array.size == 0:
        return np.zeros((0, width or 0), dtype=np.float32)
    if width is not None and array.ndim == 1:
        array = array.reshape(-1, width)
    return array.astype(np.float32, copy=False)


def safe_list(values: np.ndarray, decimals: int = 4) -> list:
    values = np.asarray(values)
    if values.size == 0:
        return []
    if np.issubdtype(values.dtype, np.floating):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.round(values.astype(np.float64), decimals)
    return values.tolist()


def normalize_prediction_by_frame(predictions: Any) -> dict[str, dict]:
    if predictions is None:
        return {}
    if isinstance(predictions, dict):
        if "frame_id" in predictions:
            predictions = [predictions]
        else:
            return {str(key): value for key, value in predictions.items()}
    result = {}
    for index, item in enumerate(predictions):
        frame_id = item.get("frame_id", item.get("frameid", item.get("sample_idx", index)))
        result[str(frame_id)] = item
    return result


def prediction_to_detections(prediction: dict | None, info: dict, score_threshold: float) -> list[dict]:
    if prediction is None:
        return []
    boxes = as_array(
        prediction.get("boxes_3d", prediction.get("boxes_lidar", prediction.get("pred_boxes", prediction.get("gt_boxes")))),
        7,
    )
    names = prediction.get("name", prediction.get("names", prediction.get("pred_names", prediction.get("gt_names"))))
    labels = prediction.get("pred_labels", prediction.get("labels"))
    scores = prediction.get("score", prediction.get("scores", prediction.get("pred_scores")))
    if scores is None:
        scores = np.ones((len(boxes),), dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    detections = []
    for det_index, box in enumerate(boxes):
        score = float(scores[det_index]) if det_index < len(scores) else 1.0
        if score < score_threshold:
            continue
        if names is not None and det_index < len(names):
            name = str(np.asarray(names, dtype=object)[det_index])
        elif labels is not None and det_index < len(labels):
            label = int(np.asarray(labels).reshape(-1)[det_index])
            name = DEFAULT_CLASSES[label - 1] if 0 < label <= len(DEFAULT_CLASSES) else str(label)
        else:
            name = "Object"
        detections.append(
            {
                "box": np.asarray(box[:7], dtype=np.float32),
                "name": name,
                "score": score,
                "source_id": f"{info['frame_id']}:{det_index}",
            }
        )
    return detections


def gt_to_detections(info: dict) -> list[dict]:
    annos = info.get("annos", {})
    boxes = as_array(annos.get("boxes_3d"), 7)
    names = np.asarray(annos.get("names", []), dtype=object)
    gt_ids = np.asarray(annos.get("gt_id", []), dtype=object)
    detections = []
    for index, box in enumerate(boxes):
        detections.append(
            {
                "box": box[:7].astype(np.float32),
                "name": str(names[index]) if index < len(names) else "Object",
                "score": 1.0,
                "source_id": str(gt_ids[index]) if index < len(gt_ids) else f"{info['frame_id']}:{index}",
            }
        )
    return detections


@dataclass
class Track:
    track_id: int
    name: str
    score: float
    state: np.ndarray
    box: np.ndarray
    hits: int = 1
    age: int = 0
    missed: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)

    def predict(self, dt: float) -> None:
        dt = max(float(dt), 0.0)
        self.state[0] += self.state[2] * dt
        self.state[1] += self.state[3] * dt
        self.box[0] = self.state[0]
        self.box[1] = self.state[1]
        self.age += 1
        self.missed += 1
        self.history.append((float(self.state[0]), float(self.state[1])))
        self.history = self.history[-20:]

    def update(self, detection: dict, dt: float) -> None:
        dt = max(float(dt), 1e-3)
        box = detection["box"].astype(np.float32)
        vx = (float(box[0]) - float(self.state[0])) / dt
        vy = (float(box[1]) - float(self.state[1])) / dt
        alpha = 0.35
        self.state[2] = (1.0 - alpha) * self.state[2] + alpha * vx
        self.state[3] = (1.0 - alpha) * self.state[3] + alpha * vy
        self.state[0] = box[0]
        self.state[1] = box[1]
        self.box = box
        self.name = detection["name"]
        self.score = float(max(self.score * 0.8, detection["score"]))
        self.hits += 1
        self.missed = 0
        self.history.append((float(box[0]), float(box[1])))
        self.history = self.history[-20:]

    @classmethod
    def start(cls, track_id: int, detection: dict) -> "Track":
        box = detection["box"].astype(np.float32)
        return cls(
            track_id=track_id,
            name=detection["name"],
            score=float(detection["score"]),
            state=np.array([box[0], box[1], 0.0, 0.0], dtype=np.float32),
            box=box.copy(),
            history=[(float(box[0]), float(box[1]))],
        )


class BevTracker:
    def __init__(self, distance_threshold: float, max_age: int, min_hits: int, max_time_gap: float):
        self.distance_threshold = distance_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_time_gap = max_time_gap
        self.tracks: list[Track] = []
        self.next_id = 1
        self.last_timestamp: float | None = None
        self.sequence_id: str | None = None

    def reset(self, sequence_id: str, timestamp: float) -> None:
        self.tracks = []
        self.last_timestamp = float(timestamp)
        self.sequence_id = sequence_id

    def step(self, sequence_id: str, timestamp: float, detections: list[dict]) -> list[Track]:
        if self.sequence_id != sequence_id or self.last_timestamp is None:
            self.reset(sequence_id, timestamp)
        raw_dt = float(timestamp) - float(self.last_timestamp)
        if self.max_time_gap > 0 and raw_dt > self.max_time_gap:
            self.reset(sequence_id, timestamp)
            raw_dt = 0.0
        dt = max(raw_dt, 0.0)
        self.last_timestamp = float(timestamp)

        for track in self.tracks:
            track.predict(dt)

        matches, unmatched_tracks, unmatched_dets = self.associate(detections)
        for track_index, det_index in matches:
            self.tracks[track_index].update(detections[det_index], dt)
        for det_index in unmatched_dets:
            self.tracks.append(Track.start(self.next_id, detections[det_index]))
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track.missed <= self.max_age]
        return [
            track
            for track in self.tracks
            if track.hits >= self.min_hits and track.missed == 0
        ]

    def associate(self, detections: list[dict]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))
        costs = np.full((len(self.tracks), len(detections)), 1e6, dtype=np.float32)
        for track_index, track in enumerate(self.tracks):
            for det_index, detection in enumerate(detections):
                if track.name != detection["name"]:
                    continue
                dx = float(track.state[0] - detection["box"][0])
                dy = float(track.state[1] - detection["box"][1])
                costs[track_index, det_index] = math.hypot(dx, dy)

        if linear_sum_assignment is not None:
            row_indices, col_indices = linear_sum_assignment(costs)
        else:
            row_indices, col_indices = greedy_assignment(costs)

        matches = []
        used_tracks = set()
        used_dets = set()
        for row, col in zip(row_indices, col_indices):
            if costs[row, col] <= self.distance_threshold:
                matches.append((int(row), int(col)))
                used_tracks.add(int(row))
                used_dets.add(int(col))
        unmatched_tracks = [index for index in range(len(self.tracks)) if index not in used_tracks]
        unmatched_dets = [index for index in range(len(detections)) if index not in used_dets]
        return matches, unmatched_tracks, unmatched_dets


def greedy_assignment(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    cols = []
    remaining = costs.copy()
    while remaining.size:
        row, col = np.unravel_index(np.argmin(remaining), remaining.shape)
        if not np.isfinite(remaining[row, col]):
            break
        rows.append(row)
        cols.append(col)
        remaining[row, :] = np.inf
        remaining[:, col] = np.inf
        if len(rows) >= min(costs.shape):
            break
    return np.asarray(rows), np.asarray(cols)


class BevRenderer:
    def __init__(self, width: int, height: int, x_range: tuple[float, float], y_range: tuple[float, float]):
        self.width = int(width)
        self.height = int(height)
        self.x_range = tuple(float(value) for value in x_range)
        self.y_range = tuple(float(value) for value in y_range)
        self.title_height = 70
        self.footer_height = 46
        self.side_margin = 54
        self.plot_left = self.side_margin
        self.plot_top = self.title_height
        self.plot_right = self.width - self.side_margin
        self.plot_bottom = self.height - self.footer_height

        plot_width = max(self.plot_right - self.plot_left, 1)
        plot_height = max(self.plot_bottom - self.plot_top, 1)
        meter_width = max(self.y_range[1] - self.y_range[0], 1e-6)
        meter_height = max(self.x_range[1] - self.x_range[0], 1e-6)
        self.scale = min(plot_width / meter_width, plot_height / meter_height)
        self.pad_x = (plot_width - meter_width * self.scale) * 0.5
        self.pad_y = (plot_height - meter_height * self.scale) * 0.5

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        u = self.plot_left + self.pad_x + (y_max - y) * self.scale
        v = self.plot_top + self.pad_y + (x_max - x) * self.scale
        return int(round(u)), int(round(v))

    def render(self, frame: dict, detections: list[dict], tracks: list[Track], draw_detections: bool) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), (18, 20, 23), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (self.width, self.title_height - 1), (27, 30, 35), -1)
        cv2.rectangle(canvas, (0, self.height - self.footer_height), (self.width, self.height), (27, 30, 35), -1)
        self.draw_grid(canvas)
        if draw_detections:
            for detection in detections:
                self.draw_box(canvas, detection["box"], COLORS["detection"], thickness=1)
        for track in tracks:
            self.draw_history(canvas, track)
            self.draw_box(canvas, track.box, COLORS["track"], thickness=2)
            u, v = self.world_to_pixel(float(track.box[0]), float(track.box[1]))
            cv2.putText(
                canvas,
                f"ID {track.track_id} {track.name} {track.score:.2f}",
                (min(max(u + 6, 8), self.width - 210), min(max(v - 8, self.title_height + 18), self.height - 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                COLORS["track"],
                1,
                cv2.LINE_AA,
            )
        self.draw_ego(canvas)
        title = (
            f"HR-4D DET+TRACK  |  {frame['sequence_id']}  |  frame {frame['frame_id'][:8]}"
        )
        cv2.putText(canvas, title, (24, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, COLORS["text"], 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"detections {len(detections):02d}   tracks {len(tracks):02d}   scale {1.0 / self.scale:.2f} m/px",
            (24, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (178, 184, 194),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Equal-scale BEV: x forward, y left positive. Smoke/exploratory visualization.",
            (24, self.height - 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (170, 176, 185),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def draw_grid(self, canvas: np.ndarray) -> None:
        plot_origin = (int(self.plot_left + self.pad_x), int(self.plot_top + self.pad_y))
        plot_end = (
            int(self.plot_left + self.pad_x + (self.y_range[1] - self.y_range[0]) * self.scale),
            int(self.plot_top + self.pad_y + (self.x_range[1] - self.x_range[0]) * self.scale),
        )
        cv2.rectangle(canvas, plot_origin, plot_end, (58, 64, 74), 1)
        for x in np.arange(math.ceil(self.x_range[0] / 20.0) * 20.0, self.x_range[1] + 1, 20.0):
            p0 = self.world_to_pixel(x, self.y_range[0])
            p1 = self.world_to_pixel(x, self.y_range[1])
            cv2.line(canvas, p0, p1, COLORS["grid"], 1)
            cv2.putText(canvas, f"{x:.0f}m", (p1[0] + 6, p1[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 136, 146), 1)
        for y in np.arange(math.ceil(self.y_range[0] / 20.0) * 20.0, self.y_range[1] + 1, 20.0):
            p0 = self.world_to_pixel(self.x_range[0], y)
            p1 = self.world_to_pixel(self.x_range[1], y)
            cv2.line(canvas, p0, p1, COLORS["grid"], 1)
            cv2.putText(canvas, f"{y:+.0f}", (p1[0] - 12, p1[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 136, 146), 1)
        p0 = self.world_to_pixel(self.x_range[0], 0.0)
        p1 = self.world_to_pixel(self.x_range[1], 0.0)
        cv2.line(canvas, p0, p1, (82, 91, 103), 1)

    def draw_ego(self, canvas: np.ndarray) -> None:
        u, v = self.world_to_pixel(0.0, 0.0)
        triangle = np.array([[u, v - 13], [u - 8, v + 10], [u + 8, v + 10]], dtype=np.int32)
        cv2.fillConvexPoly(canvas, triangle, COLORS["ego"])
        cv2.putText(canvas, "ego", (u + 10, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["ego"], 1, cv2.LINE_AA)

    def draw_box(self, canvas: np.ndarray, box: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        corners = box_corners_bev(box)
        pixels = np.asarray([self.world_to_pixel(x, y) for x, y in corners], dtype=np.int32)
        cv2.polylines(canvas, [pixels], True, color, thickness, cv2.LINE_AA)
        x, y, _, length, _, _, yaw = [float(value) for value in box[:7]]
        front = (x + math.cos(yaw) * length * 0.5, y + math.sin(yaw) * length * 0.5)
        cv2.line(canvas, self.world_to_pixel(x, y), self.world_to_pixel(*front), color, thickness, cv2.LINE_AA)

    def draw_history(self, canvas: np.ndarray, track: Track) -> None:
        if len(track.history) < 2:
            return
        points = np.asarray([self.world_to_pixel(x, y) for x, y in track.history], dtype=np.int32)
        cv2.polylines(canvas, [points], False, (45, 170, 190), 1, cv2.LINE_AA)


def box_corners_bev(box: np.ndarray) -> np.ndarray:
    x, y, _, length, width, _, yaw = [float(value) for value in box[:7]]
    local = np.array(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        dtype=np.float32,
    )
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]], dtype=np.float32)
    return local @ rotation.T + np.array([x, y], dtype=np.float32)


def build_overlay_frame(info: dict, detections: list[dict], tracks: list[Track]) -> dict:
    return {
        "sequence_id": str(info["sequence_id"]),
        "frame_id": str(info["frame_id"]),
        "timestamp": float(info["timestamp"]),
        "detections": [
            {
                "name": det["name"],
                "score": round(float(det["score"]), 4),
                "box": safe_list(det["box"]),
                "source_id": det["source_id"],
            }
            for det in detections
        ],
        "tracks": [
            {
                "track_id": int(track.track_id),
                "name": track.name,
                "score": round(float(track.score), 4),
                "age": int(track.age),
                "hits": int(track.hits),
                "box": safe_list(track.box),
                "velocity": safe_list(track.state[2:4], 4),
                "history": [[round(x, 4), round(y, 4)] for x, y in track.history],
            }
            for track in tracks
        ],
    }


def build_tracked_prediction(info: dict, tracks: list[Track]) -> dict:
    boxes = np.asarray([track.box for track in tracks], dtype=np.float32).reshape(-1, 7)
    scores = np.asarray([track.score for track in tracks], dtype=np.float32)
    names = np.asarray([track.name for track in tracks], dtype=object)
    track_ids = np.asarray([track.track_id for track in tracks], dtype=np.int32)
    return {
        "frame_id": str(info["frame_id"]),
        "sequence_id": str(info["sequence_id"]),
        "timestamp": float(info["timestamp"]),
        "name": names,
        "boxes_3d": boxes,
        "score": scores,
        "track_id": track_ids,
    }


def make_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else output_dir / "hr4d_tracks_overlay.json"
    output_pkl = Path(args.output_pkl) if args.output_pkl else output_dir / "hr4d_tracks.pkl"
    output_video = Path(args.output_video) if args.output_video else None
    for path in (output_json, output_pkl, output_video):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    return output_json, output_pkl, output_video


def main() -> None:
    args = parse_args()
    if not args.predictions and not args.use_gt_as_detections:
        raise SystemExit("Provide --predictions result.pkl or pass --use-gt-as-detections for smoke testing.")

    infos = load_pickle(args.infos)
    if args.max_frames > 0:
        infos = infos[: args.max_frames]
    predictions = normalize_prediction_by_frame(load_pickle(args.predictions)) if args.predictions else {}

    output_json, output_pkl, output_video = make_output_paths(args)
    tracker = BevTracker(args.distance_threshold, args.max_age, args.min_hits, args.max_time_gap)
    renderer = BevRenderer(args.bev_width, args.bev_height, tuple(args.x_range), tuple(args.y_range))
    video_writer = None
    if output_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_video), fourcc, args.fps, (args.bev_width, args.bev_height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Unable to open video writer: {output_video}")

    overlay_frames = []
    tracked_predictions = []
    total_detections = 0
    total_tracks = 0
    for info in infos:
        if args.use_gt_as_detections:
            detections = gt_to_detections(info)
        else:
            prediction = predictions.get(str(info["frame_id"]))
            detections = prediction_to_detections(prediction, info, args.score_threshold)
        tracks = tracker.step(str(info["sequence_id"]), float(info["timestamp"]), detections)
        overlay_frames.append(build_overlay_frame(info, detections, tracks))
        tracked_predictions.append(build_tracked_prediction(info, tracks))
        total_detections += len(detections)
        total_tracks += len(tracks)
        if video_writer is not None:
            video_writer.write(renderer.render(info, detections, tracks, args.draw_detections))

    if video_writer is not None:
        video_writer.release()

    overlay = {
        "schema": "hr4d_tracking_overlay_v1",
        "evidence_label": "smoke" if args.use_gt_as_detections else "exploratory",
        "infos": str(args.infos),
        "predictions": str(args.predictions) if args.predictions else None,
        "use_gt_as_detections": bool(args.use_gt_as_detections),
        "tracker": {
            "type": "constant_velocity_bev",
            "distance_threshold": args.distance_threshold,
            "max_age": args.max_age,
            "max_time_gap": args.max_time_gap,
            "min_hits": args.min_hits,
            "score_threshold": args.score_threshold,
            "sequence_boundary_reset": True,
        },
        "summary": {
            "frames": len(overlay_frames),
            "detections": total_detections,
            "emitted_tracks": total_tracks,
            "avg_detections_per_frame": total_detections / max(len(overlay_frames), 1),
            "avg_tracks_per_frame": total_tracks / max(len(overlay_frames), 1),
        },
        "frames": overlay_frames,
    }
    output_json.write_text(json.dumps(overlay, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_pickle(output_pkl, tracked_predictions)

    print(json.dumps(overlay["summary"], ensure_ascii=False))
    print(f"wrote_json={output_json}")
    print(f"wrote_pkl={output_pkl}")
    if output_video is not None:
        print(f"wrote_video={output_video}")


if __name__ == "__main__":
    main()
