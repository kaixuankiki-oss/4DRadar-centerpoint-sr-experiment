#!/usr/bin/env python3
"""Frame-level evaluation diff utilities for HR-4D visualization."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pcdet.datasets.hr4d.hr4d_eval.evaluation import (
    HR4DEvalConfig,
    elliptical_center_distance,
    evaluation_region_mask,
)


CLASS_MAPPING = {
    "Car": "Vehicle",
    "Truck": "Vehicle",
    "Bus": "Vehicle",
    "Split_vehicle": "Vehicle",
    "Vehicle_attachment": "Vehicle",
    "Vehicle": "Vehicle",
    "Tricycle": "Cyclist",
    "Cyclist": "Cyclist",
    "Pedestrian": "Pedestrian",
}


@dataclass(frozen=True)
class EvalDiffConfig:
    score_threshold: float = 0.1
    match_lateral_threshold: float = 2.0
    loc_warning_threshold: float = 1.0
    radial_scale: float = 2.0
    max_cases: int = 80
    max_frames: int = 40


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def canonical_class(name: Any) -> str:
    return CLASS_MAPPING.get(str(name), str(name))


def normalize_prediction_by_frame(predictions: Any) -> dict[str, dict]:
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


def as_boxes(values: Any) -> np.ndarray:
    if values is None:
        return np.empty((0, 7), dtype=np.float64)
    boxes = np.asarray(values, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 7), dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes.reshape(-1, 7)
    if boxes.shape[1] < 7:
        raise ValueError(f"boxes must have shape (N, >=7), got {boxes.shape}")
    return boxes[:, :7]


def first_existing(payload: dict, keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


def normalize_gt(info: dict) -> list[dict]:
    annos = info.get("annos", info)
    names = np.asarray(first_existing(annos, ("names", "name", "gt_names"), []), dtype=object).reshape(-1)
    boxes = as_boxes(first_existing(annos, ("boxes_3d", "gt_boxes_lidar", "gt_boxes", "boxes_lidar"), None))
    ids = np.asarray(annos.get("gt_id", [f"gt-{i}" for i in range(len(boxes))]), dtype=object).reshape(-1)
    region_mask = evaluation_region_mask(boxes, HR4DEvalConfig()) if len(boxes) else np.zeros(0, dtype=bool)
    result = []
    for index, box in enumerate(boxes):
        name = str(names[index]) if index < len(names) else "Object"
        result.append(
            {
                "gt_index": index,
                "gt_id": str(ids[index]) if index < len(ids) else f"gt-{index}",
                "name": name,
                "class_name": canonical_class(name),
                "box": clean_list(box),
                "distance": float(np.linalg.norm(box[:2])),
                "in_region": bool(region_mask[index]),
            }
        )
    return result


def normalize_predictions(prediction: dict | None, score_threshold: float) -> list[dict]:
    if prediction is None:
        return []
    boxes = as_boxes(first_existing(prediction, ("boxes_3d", "boxes_lidar", "pred_boxes", "gt_boxes"), None))
    names = first_existing(prediction, ("name", "names", "pred_names", "gt_names"), None)
    labels = first_existing(prediction, ("pred_labels", "labels"), None)
    scores = first_existing(prediction, ("score", "scores", "pred_scores"), None)
    if scores is None:
        scores = np.ones(len(boxes), dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    names = np.asarray(names, dtype=object).reshape(-1) if names is not None else None
    labels = np.asarray(labels).reshape(-1) if labels is not None else None
    region_mask = evaluation_region_mask(boxes, HR4DEvalConfig()) if len(boxes) else np.zeros(0, dtype=bool)

    result = []
    for index, box in enumerate(boxes):
        score = float(scores[index]) if index < len(scores) else 1.0
        if score < score_threshold:
            continue
        if names is not None and index < len(names):
            name = str(names[index])
        elif labels is not None and index < len(labels):
            name = str(int(labels[index]))
        else:
            name = "Object"
        result.append(
            {
                "pred_index": index,
                "name": name,
                "class_name": canonical_class(name),
                "box": clean_list(box),
                "score": score,
                "distance": float(np.linalg.norm(box[:2])),
                "in_region": bool(region_mask[index]),
            }
        )
    return result


def match_frame(
    info: dict,
    prediction: dict | None,
    frame_index: int,
    config: EvalDiffConfig | None = None,
) -> dict:
    config = config or EvalDiffConfig()
    gts = normalize_gt(info)
    preds = normalize_predictions(prediction, config.score_threshold)
    active_gt_indices = [i for i, item in enumerate(gts) if item["in_region"]]
    active_pred_indices = [i for i, item in enumerate(preds) if item["in_region"]]
    taken_gt: set[int] = set()
    pred_status: dict[int, dict] = {}
    gt_status: dict[int, dict] = {}
    matches = []
    cases = []

    sorted_preds = sorted(active_pred_indices, key=lambda i: preds[i]["score"], reverse=True)
    for pred_list_index in sorted_preds:
        pred = preds[pred_list_index]
        candidates = [
            gt_list_index for gt_list_index in active_gt_indices
            if gt_list_index not in taken_gt and gts[gt_list_index]["class_name"] == pred["class_name"]
        ]
        best_gt = None
        best_distance = None
        if candidates:
            distances = np.asarray([
                elliptical_center_distance(gts[i]["box"][:2], np.asarray(pred["box"][:2], dtype=np.float64)[None, :2], config.radial_scale)[0]
                for i in candidates
            ])
            best_pos = int(np.argmin(distances))
            if float(distances[best_pos]) <= config.match_lateral_threshold:
                best_gt = candidates[best_pos]
                best_distance = float(distances[best_pos])

        if best_gt is None:
            continue

        taken_gt.add(best_gt)
        errors = box_errors(gts[best_gt]["box"], pred["box"], config.radial_scale)
        match_id = f"f{frame_index:06d}-m{len(matches):03d}"
        match = {
            "eval_id": match_id,
            "type": "TP",
            "frame_index": frame_index,
            "frame_id": str(info.get("frame_id", frame_index)),
            "gt_index": gts[best_gt]["gt_index"],
            "pred_index": pred["pred_index"],
            "class_name": pred["class_name"],
            "score": pred["score"],
            "equivalent_lateral_error": best_distance,
            **errors,
        }
        matches.append(match)
        pred_status[pred_list_index] = {"match_status": "tp", "eval_id": match_id, "gt_index": gts[best_gt]["gt_index"]}
        gt_status[best_gt] = {"match_status": "tp", "eval_id": match_id, "pred_index": pred["pred_index"]}
        if best_distance >= config.loc_warning_threshold:
            cases.append(
                {
                    **match,
                    "type": "LOC",
                    "reason": "large_center_error",
                    "case_score": round(50.0 + 20.0 * best_distance + 5.0 * errors["orient_error"], 4),
                }
            )

    for pred_list_index in active_pred_indices:
        if pred_list_index in pred_status:
            continue
        pred = preds[pred_list_index]
        reason, nearest = classify_fp(pred, gts, matches, config)
        case = {
            "eval_id": f"f{frame_index:06d}-fp{len(cases):03d}",
            "type": "FP",
            "reason": reason,
            "frame_index": frame_index,
            "frame_id": str(info.get("frame_id", frame_index)),
            "pred_index": pred["pred_index"],
            "class_name": pred["class_name"],
            "score": pred["score"],
            "nearest_gt_index": nearest.get("gt_index"),
            "nearest_gt_class": nearest.get("class_name"),
            "nearest_distance": nearest.get("distance"),
            "case_score": round(80.0 + 20.0 * pred["score"], 4),
        }
        cases.append(case)
        pred_status[pred_list_index] = {"match_status": "fp", "eval_id": case["eval_id"], "reason": reason}

    for gt_list_index in active_gt_indices:
        if gt_list_index in gt_status:
            continue
        gt = gts[gt_list_index]
        reason, nearest = classify_fn(gt, preds, config)
        case = {
            "eval_id": f"f{frame_index:06d}-fn{len(cases):03d}",
            "type": "FN",
            "reason": reason,
            "frame_index": frame_index,
            "frame_id": str(info.get("frame_id", frame_index)),
            "gt_index": gt["gt_index"],
            "gt_id": gt["gt_id"],
            "class_name": gt["class_name"],
            "nearest_pred_index": nearest.get("pred_index"),
            "nearest_pred_class": nearest.get("class_name"),
            "nearest_score": nearest.get("score"),
            "nearest_distance": nearest.get("distance"),
            "case_score": round(110.0 + min(gt["distance"], 200.0) / 10.0, 4),
        }
        cases.append(case)
        gt_status[gt_list_index] = {"match_status": "fn", "eval_id": case["eval_id"], "reason": reason}

    for pred_list_index, status in pred_status.items():
        preds[pred_list_index].update(status)
    for gt_list_index, status in gt_status.items():
        gts[gt_list_index].update(status)

    summary = {
        "tp": len(matches),
        "fp": sum(1 for case in cases if case["type"] == "FP"),
        "fn": sum(1 for case in cases if case["type"] == "FN"),
        "loc": sum(1 for case in cases if case["type"] == "LOC"),
        "gt": len(active_gt_indices),
        "pred": len(active_pred_indices),
        "score": round(sum(case["case_score"] for case in cases), 4),
    }
    return {
        "frame_index": frame_index,
        "frame_id": str(info.get("frame_id", frame_index)),
        "sequence_id": str(info.get("sequence_id", "")),
        "timestamp": float(info.get("timestamp", 0.0)),
        "summary": summary,
        "gt": gts,
        "predictions": preds,
        "matches": matches,
        "cases": sorted(cases, key=lambda item: item["case_score"], reverse=True),
    }


def classify_fp(pred: dict, gts: list[dict], matches: list[dict], config: EvalDiffConfig) -> tuple[str, dict]:
    nearest = nearest_gt(pred, gts, config)
    if nearest and nearest["distance"] <= config.match_lateral_threshold and nearest["class_name"] != pred["class_name"]:
        return "class_confusion", nearest
    matched_gt_indices = {match["gt_index"] for match in matches}
    if nearest and nearest["gt_index"] in matched_gt_indices and nearest["distance"] <= config.match_lateral_threshold:
        return "duplicate", nearest
    if nearest and nearest["distance"] <= config.match_lateral_threshold * 2.0:
        return "localization_or_class_boundary", nearest
    return "background_or_out_of_roi_context", nearest or {}


def classify_fn(gt: dict, preds: list[dict], config: EvalDiffConfig) -> tuple[str, dict]:
    nearest = nearest_pred(gt, preds, config)
    if nearest and nearest["distance"] <= config.match_lateral_threshold and nearest["class_name"] != gt["class_name"]:
        return "class_confusion", nearest
    if nearest and nearest["class_name"] == gt["class_name"] and nearest["distance"] <= config.match_lateral_threshold * 2.0:
        return "localization_miss", nearest
    return "missed_detection", nearest or {}


def nearest_gt(pred: dict, gts: list[dict], config: EvalDiffConfig) -> dict:
    candidates = [gt for gt in gts if gt["in_region"]]
    return nearest_box(pred, candidates, "gt", config)


def nearest_pred(gt: dict, preds: list[dict], config: EvalDiffConfig) -> dict:
    candidates = [pred for pred in preds if pred["in_region"]]
    return nearest_box(gt, candidates, "pred", config)


def nearest_box(item: dict, candidates: list[dict], prefix: str, config: EvalDiffConfig) -> dict:
    if not candidates:
        return {}
    distances = [
        float(elliptical_center_distance(candidate["box"][:2], np.asarray(item["box"][:2], dtype=np.float64)[None, :2], config.radial_scale)[0])
        for candidate in candidates
    ]
    index = int(np.argmin(distances))
    candidate = candidates[index]
    id_key = "gt_index" if prefix == "gt" else "pred_index"
    return {
        id_key: candidate[id_key],
        "class_name": candidate["class_name"],
        "score": candidate.get("score"),
        "distance": distances[index],
    }


def box_errors(gt_box: list[float], pred_box: list[float], radial_scale: float) -> dict:
    gt = np.asarray(gt_box, dtype=np.float64)
    pred = np.asarray(pred_box, dtype=np.float64)
    gt_range = np.linalg.norm(gt[:2])
    radial = gt[:2] / gt_range if gt_range > 1e-8 else np.array([1.0, 0.0])
    lateral = np.array([-radial[1], radial[0]])
    delta = pred[:2] - gt[:2]
    radial_error = float(delta @ radial)
    lateral_error = float(delta @ lateral)
    equivalent = float(math.sqrt((radial_error / radial_scale) ** 2 + lateral_error ** 2))
    return {
        "trans_error": float(np.linalg.norm(delta)),
        "radial_error": radial_error,
        "lateral_error": lateral_error,
        "equivalent_lateral_error": equivalent,
        "scale_error": scale_error(gt[3:6], pred[3:6]),
        "orient_error": yaw_error(gt[6], pred[6]),
    }


def yaw_error(gt_yaw: float, pred_yaw: float) -> float:
    return float(abs((pred_yaw - gt_yaw + np.pi) % (2 * np.pi) - np.pi))


def scale_error(gt_size: np.ndarray, pred_size: np.ndarray) -> float:
    gt_size = np.maximum(np.asarray(gt_size, dtype=np.float64), 1e-8)
    pred_size = np.maximum(np.asarray(pred_size, dtype=np.float64), 1e-8)
    return float(1.0 - np.prod(np.minimum(gt_size, pred_size)) / np.prod(np.maximum(gt_size, pred_size)))


def build_eval_report(infos: list[dict], predictions: Any, config: EvalDiffConfig | None = None) -> dict:
    config = config or EvalDiffConfig()
    pred_by_frame = normalize_prediction_by_frame(predictions)
    frames = [
        match_frame(info, pred_by_frame.get(str(info.get("frame_id"))), index, config)
        for index, info in enumerate(infos)
    ]
    cases = [case for frame in frames for case in frame["cases"]]
    cases.sort(key=lambda item: item["case_score"], reverse=True)
    summary = {
        "frames": len(frames),
        "tp": sum(frame["summary"]["tp"] for frame in frames),
        "fp": sum(frame["summary"]["fp"] for frame in frames),
        "fn": sum(frame["summary"]["fn"] for frame in frames),
        "loc": sum(frame["summary"]["loc"] for frame in frames),
        "gt": sum(frame["summary"]["gt"] for frame in frames),
        "pred": sum(frame["summary"]["pred"] for frame in frames),
    }
    return {"config": config.__dict__, "summary": summary, "frames": frames, "cases": cases}


def select_review_frames(report: dict, max_cases: int, max_frames: int) -> tuple[list[dict], list[int]]:
    selected_cases = report["cases"][:max_cases] if max_cases > 0 else report["cases"]
    frame_indices = []
    for case in selected_cases:
        if case["frame_index"] not in frame_indices:
            frame_indices.append(case["frame_index"])
        if max_frames > 0 and len(frame_indices) >= max_frames:
            break
    if not frame_indices and report["frames"]:
        frame_indices = [0]
    selected_cases = [case for case in selected_cases if case["frame_index"] in set(frame_indices)]
    return selected_cases, frame_indices


def overlay_frames_from_report(report: dict) -> dict:
    frames = {}
    for frame in report["frames"]:
        objects = []
        for pred in frame["predictions"]:
            if not pred["in_region"]:
                continue
            status = pred.get("match_status", "unmatched")
            objects.append(
                {
                    "name": pred["name"],
                    "class_name": pred["class_name"],
                    "score": pred["score"],
                    "box": pred["box"],
                    "source": "pred",
                    "match_status": status,
                    "eval_id": pred.get("eval_id", ""),
                    "pred_index": pred["pred_index"],
                    "gt_index": pred.get("gt_index"),
                    "reason": pred.get("reason", ""),
                }
            )
        frames[frame["frame_id"]] = objects
    return {"schema": "hr4d_eval_overlay_v1", "frames": frames}


def clean_list(values: Any, decimals: int = 5) -> list:
    array = np.asarray(values)
    if array.size == 0:
        return []
    array = np.nan_to_num(array.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.round(array, decimals).tolist()
