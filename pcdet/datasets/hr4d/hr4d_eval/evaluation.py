"""nuScenes-style center-distance evaluation for the HR-4D project."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def make_distance_bins(eval_range: Sequence[float], step: float = 50.0) -> Tuple[Tuple[float, float], ...]:
    """Build contiguous distance bins for the HR-4D evaluator."""

    if len(eval_range) != 2:
        raise ValueError('eval_range must be [start, end]')
    start, end = (float(eval_range[0]), float(eval_range[1]))
    if start != 0.0 or end <= start:
        raise ValueError('eval_range must be a positive interval starting at 0')
    if step <= 0:
        raise ValueError('distance bin step must be positive')

    bins = []
    current = start
    while current < end:
        next_end = min(current + step, end)
        bins.append((current, next_end))
        current = next_end
    return tuple(bins)


@dataclass(frozen=True)
class HR4DEvalConfig:
    """Configuration for the HR-4D detection protocol.

    ``lateral_thresholds`` are the ellipse minor-axis thresholds in meters.
    The major axis points from the ego origin to each GT center and is
    ``radial_scale`` times larger for RADAR. LIDAR uses equal radial/lateral
    center-distance weighting.
    """

    lateral_thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    radial_scale: float = 2.0
    sensor_type: str = 'RADAR'
    distance_bins: Tuple[Tuple[float, float], ...] = (
        (0.0, 50.0),
        (50.0, 100.0),
        (100.0, 150.0),
        (150.0, 200.0),
    )
    max_forward: float = 200.0
    max_lateral: float = 20.0
    max_abs_angle_deg: float = 40.0
    min_recall: float = 0.1
    min_precision: float = 0.1
    num_recall_points: int = 101
    tp_lateral_threshold: float = 2.0
    vehicle_dynamic_distance_range: Tuple[float, float] = (100.0, 200.0)
    vehicle_dynamic_class_names: Tuple[str, ...] = (
        'Vehicle',
        'Car',
        'LargeV',
        'LargeVehicle',
        'Truck',
        'Bus',
        'Split_vehicle',
        'Split_Vehicle',
        'Vehicle_attachment',
    )
    dynamic_speed_threshold_mps: float = 1.0

    @classmethod
    def from_ranges(
            cls,
            sensor_type: str = 'RADAR',
            eval_range: Sequence[float] = (0.0, 200.0),
            far_dynamic_vehicle: Sequence[float] = (100.0, 200.0),
    ) -> 'HR4DEvalConfig':
        if len(far_dynamic_vehicle) != 2:
            raise ValueError('far_dynamic_vehicle must be [start, end]')
        distance_bins = make_distance_bins(eval_range)
        return cls(
            sensor_type=str(sensor_type).upper(),
            distance_bins=distance_bins,
            max_forward=float(eval_range[1]),
            vehicle_dynamic_distance_range=(
                float(far_dynamic_vehicle[0]),
                float(far_dynamic_vehicle[1]),
            ),
        )

    @property
    def normalized_sensor_type(self) -> str:
        return str(self.sensor_type).upper()

    @property
    def matching_radial_scale(self) -> float:
        if self.normalized_sensor_type == 'LIDAR':
            return 1.0
        return self.radial_scale

    def __post_init__(self):
        if not self.lateral_thresholds or any(x <= 0 for x in self.lateral_thresholds):
            raise ValueError('lateral_thresholds must contain positive values')
        if self.radial_scale <= 0:
            raise ValueError('radial_scale must be positive')
        if self.normalized_sensor_type not in ('RADAR', 'LIDAR'):
            raise ValueError('sensor_type must be RADAR or LIDAR')
        if self.max_forward <= 0 or self.max_lateral <= 0:
            raise ValueError('evaluation bounds must be positive')
        if not 0 < self.max_abs_angle_deg <= 90:
            raise ValueError('max_abs_angle_deg must be in (0, 90]')
        if not 0 <= self.min_recall < 1 or not 0 <= self.min_precision < 1:
            raise ValueError('min_recall and min_precision must be in [0, 1)')
        if self.num_recall_points < 2:
            raise ValueError('num_recall_points must be at least 2')
        if self.tp_lateral_threshold not in self.lateral_thresholds:
            raise ValueError('tp_lateral_threshold must be included in lateral_thresholds')
        if not self.distance_bins or self.distance_bins[0][0] != 0.0:
            raise ValueError('distance bins must start at zero')
        dyn_start, dyn_end = self.vehicle_dynamic_distance_range
        if dyn_start < 0 or dyn_end <= dyn_start:
            raise ValueError('vehicle_dynamic_distance_range must be a positive interval')
        if self.dynamic_speed_threshold_mps < 0:
            raise ValueError('dynamic_speed_threshold_mps must be non-negative')
        previous_end = None
        for start, end in self.distance_bins:
            if start < 0 or end <= start:
                raise ValueError('distance bins must be positive increasing intervals')
            if previous_end is not None and start != previous_end:
                raise ValueError('distance bins must be contiguous')
            previous_end = end


@dataclass
class _FrameAnnotations:
    names: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    dynamic: Optional[np.ndarray] = None
    frame_id: Optional[str] = None


def elliptical_center_distance(
        gt_xy: Sequence[float],
        pred_xy: np.ndarray,
        radial_scale: float = 2.0,
) -> np.ndarray:
    """Return equivalent lateral distance in the GT-aligned BEV ellipse.

    A returned value of ``t`` lies on an ellipse with lateral radius ``t`` and
    radial radius ``radial_scale * t``. Only XY center coordinates are used.
    """

    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(2)
    pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    gt_range = np.linalg.norm(gt_xy)
    radial_unit = gt_xy / gt_range if gt_range > 1e-8 else np.array([1.0, 0.0])
    lateral_unit = np.array([-radial_unit[1], radial_unit[0]])
    delta = pred_xy - gt_xy
    radial_error = delta @ radial_unit
    lateral_error = delta @ lateral_unit
    return np.sqrt((radial_error / radial_scale) ** 2 + lateral_error ** 2)


def evaluation_region_mask(boxes: np.ndarray, config: HR4DEvalConfig) -> np.ndarray:
    """Filter boxes to the shared forward rectangular and fan-shaped ROI."""

    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return np.zeros(0, dtype=bool)
    x, y = boxes[:, 0], boxes[:, 1]
    distance = np.linalg.norm(boxes[:, :2], axis=1)
    angle_deg = np.degrees(np.abs(np.arctan2(y, x)))
    return (
        (x >= 0.0)
        & (x <= config.max_forward)
        & (np.abs(y) <= config.max_lateral)
        & (distance <= config.distance_bins[-1][1])
        & (angle_deg <= config.max_abs_angle_deg)
    )


def _first_existing(anno: dict, keys: Sequence[str], label: str):
    for key in keys:
        if key in anno:
            return anno[key]
    raise KeyError(f'missing {label}; expected one of {tuple(keys)}')


def _optional_array(anno: dict, keys: Sequence[str], length: int):
    for key in keys:
        if key not in anno:
            continue
        values = np.asarray(anno[key])
        if len(values) != length:
            raise ValueError(f'{key} has {len(values)} entries, expected {length}')
        return values
    return None


def _dynamic_mask_from_annotation(anno: dict, boxes: np.ndarray, config: HR4DEvalConfig) -> Optional[np.ndarray]:
    """Return GT dynamic mask when sidecar velocity/dynamic fields are available."""

    dynamic = _optional_array(
        anno,
        ('dynamic_gt_candidate', 'is_dynamic', 'dynamic'),
        len(boxes),
    )
    if dynamic is not None:
        return dynamic.astype(bool).reshape(-1)

    speed = _optional_array(
        anno,
        ('speed_abs_xy_candidate', 'speed_abs_xy', 'gt_speed_abs_xy'),
        len(boxes),
    )
    if speed is not None:
        return np.asarray(speed, dtype=np.float64).reshape(-1) >= config.dynamic_speed_threshold_mps

    velocity = _optional_array(
        anno,
        ('gt_velocity_xy_from_radar_candidate', 'velocity_xy_candidate'),
        len(boxes),
    )
    if velocity is not None:
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.shape != (len(boxes), 2):
            raise ValueError(
                'gt_velocity_xy_from_radar_candidate must have shape (N, 2)'
            )
        valid = np.isfinite(velocity).all(axis=1)
        speed_xy = np.linalg.norm(velocity, axis=1)
        return valid & (speed_xy >= config.dynamic_speed_threshold_mps)

    if boxes.shape[1] >= 9:
        velocity = boxes[:, 7:9]
        valid = np.isfinite(velocity).all(axis=1)
        speed_xy = np.linalg.norm(velocity, axis=1)
        return valid & (speed_xy >= config.dynamic_speed_threshold_mps)

    return None


def _normalize_annotations(
        annos: Sequence[dict],
        is_prediction: bool,
        config: HR4DEvalConfig,
) -> List[_FrameAnnotations]:
    normalized = []
    for frame_index, anno in enumerate(annos):
        frame_id = anno.get('frame_id', None)
        if frame_id is not None:
            frame_id = str(frame_id)
        names = np.asarray(
            _first_existing(anno, ('name', 'names', 'gt_names'), 'class names'),
            dtype=object,
        ).reshape(-1)
        boxes = np.asarray(
            _first_existing(
                anno,
                ('boxes_3d', 'boxes_lidar', 'gt_boxes_lidar', 'gt_boxes'),
                '3D boxes',
            ),
            dtype=np.float64,
        )
        if boxes.size == 0:
            boxes = np.empty((0, 7), dtype=np.float64)
        elif boxes.ndim != 2 or boxes.shape[1] < 7:
            raise ValueError(f'frame {frame_index}: boxes must have shape (N, >=7)')
        if len(names) != len(boxes):
            raise ValueError(f'frame {frame_index}: names and boxes have different lengths')
        if boxes.shape[1] == 7:
            for velocity_key in ('vels', 'velocity', 'velocities'):
                if velocity_key in anno:
                    velocities = np.asarray(anno[velocity_key], dtype=np.float64)
                    if velocities.shape != (len(boxes), 2):
                        raise ValueError(f'frame {frame_index}: velocities must have shape (N, 2)')
                    boxes = np.concatenate([boxes, velocities], axis=1)
                    break

        if is_prediction:
            scores = np.asarray(
                _first_existing(anno, ('score', 'scores', 'pred_scores'), 'prediction scores'),
                dtype=np.float64,
            ).reshape(-1)
            if len(scores) != len(boxes):
                raise ValueError(f'frame {frame_index}: scores and boxes have different lengths')
            if not np.isfinite(scores).all():
                raise ValueError(f'frame {frame_index}: prediction scores must be finite')
        else:
            scores = np.ones(len(boxes), dtype=np.float64)
        dynamic = None if is_prediction else _dynamic_mask_from_annotation(anno, boxes, config)
        normalized.append(_FrameAnnotations(
            names=names, boxes=boxes, scores=scores, dynamic=dynamic, frame_id=frame_id
        ))
    return normalized


def _annotation_frame_id(anno: dict) -> Optional[str]:
    frame_id = anno.get('frame_id', None)
    if frame_id is None:
        return None
    return str(frame_id)


def _align_annotations_by_frame_id(
        gt_annos: Sequence[dict],
        pred_annos: Sequence[dict],
) -> Tuple[List[dict], List[dict]]:
    gt_frame_ids = [_annotation_frame_id(anno) for anno in gt_annos]
    pred_frame_ids = [_annotation_frame_id(anno) for anno in pred_annos]
    if not all(gt_frame_ids) or not all(pred_frame_ids):
        if len(gt_annos) != len(pred_annos):
            raise ValueError('the number of GT frames must match prediction frames')
        return list(gt_annos), list(pred_annos)

    gt_by_frame_id: Dict[str, dict] = {}
    for frame_id, anno in zip(gt_frame_ids, gt_annos):
        if frame_id in gt_by_frame_id:
            raise ValueError('duplicate GT frame_id: %s' % frame_id)
        gt_by_frame_id[frame_id] = anno

    aligned_gt_annos, aligned_pred_annos = [], []
    seen_pred_frame_ids = set()
    for frame_id, anno in zip(pred_frame_ids, pred_annos):
        if frame_id in seen_pred_frame_ids:
            continue
        seen_pred_frame_ids.add(frame_id)
        gt_anno = gt_by_frame_id.get(frame_id)
        if gt_anno is None:
            continue
        aligned_gt_annos.append(gt_anno)
        aligned_pred_annos.append(anno)

    if not aligned_gt_annos and gt_annos:
        raise ValueError('no prediction frames match GT frame_id')
    return aligned_gt_annos, aligned_pred_annos


def _filter_region(frames: Sequence[_FrameAnnotations], config: HR4DEvalConfig) -> List[_FrameAnnotations]:
    filtered = []
    for frame in frames:
        mask = evaluation_region_mask(frame.boxes, config)
        filtered.append(_FrameAnnotations(
            names=frame.names[mask],
            boxes=frame.boxes[mask],
            scores=frame.scores[mask],
            dynamic=frame.dynamic[mask] if frame.dynamic is not None else None,
            frame_id=frame.frame_id,
        ))
    return filtered


def _segment_name(distance_range: Optional[Tuple[float, float]]) -> str:
    if distance_range is None:
        return 'overall'
    start, end = distance_range
    return f'{start:g}-{end:g}m'


def _range_key(distance_range: Sequence[float]) -> str:
    start, end = distance_range
    return f'{start:g}_{end:g}'.replace('.', 'p')


def _distance_mask(
        boxes: np.ndarray,
        distance_range: Optional[Tuple[float, float]],
        include_upper: bool = False,
) -> np.ndarray:
    if distance_range is None:
        return np.ones(len(boxes), dtype=bool)
    start, end = distance_range
    distance = np.linalg.norm(boxes[:, :2], axis=1)
    if include_upper:
        return (distance >= start) & (distance <= end)
    return (distance >= start) & (distance < end)


def _class_segment_data(
        frames: Sequence[_FrameAnnotations],
        class_name: str,
        distance_range: Optional[Tuple[float, float]],
        include_upper: bool = False,
) -> List[_FrameAnnotations]:
    output = []
    for frame in frames:
        mask = (frame.names == class_name) & _distance_mask(
            frame.boxes, distance_range, include_upper=include_upper
        )
        output.append(_FrameAnnotations(
            names=frame.names[mask],
            boxes=frame.boxes[mask],
            scores=frame.scores[mask],
            dynamic=frame.dynamic[mask] if frame.dynamic is not None else None,
            frame_id=frame.frame_id,
        ))
    return output


def _vehicle_dynamic_data(
        frames: Sequence[_FrameAnnotations],
        config: HR4DEvalConfig,
        is_gt: bool,
) -> Optional[List[_FrameAnnotations]]:
    """Build the configured dynamic Vehicle-only slice with vehicle classes merged."""

    if is_gt and not any(frame.dynamic is not None for frame in frames):
        return None

    vehicle_classes = np.asarray(config.vehicle_dynamic_class_names, dtype=object)
    output = []
    for frame in frames:
        mask = (
            np.isin(frame.names.astype(object), vehicle_classes)
            & _distance_mask(
                frame.boxes,
                config.vehicle_dynamic_distance_range,
                include_upper=True,
            )
        )
        if is_gt:
            if frame.dynamic is None:
                dynamic_mask = np.zeros(len(frame.names), dtype=bool)
            else:
                dynamic_mask = frame.dynamic.astype(bool)
            mask &= dynamic_mask

        count = int(np.count_nonzero(mask))
        output.append(_FrameAnnotations(
            names=np.asarray(['Vehicle'] * count, dtype=object),
            boxes=frame.boxes[mask],
            scores=frame.scores[mask],
            dynamic=frame.dynamic[mask] if frame.dynamic is not None else None,
            frame_id=frame.frame_id,
        ))
    return output


def _yaw_error(gt_yaw: float, pred_yaw: float) -> float:
    return float(abs((pred_yaw - gt_yaw + np.pi) % (2 * np.pi) - np.pi))


def _scale_error(gt_size: np.ndarray, pred_size: np.ndarray) -> float:
    gt_size = np.maximum(gt_size, 1e-8)
    pred_size = np.maximum(pred_size, 1e-8)
    scale_iou = np.prod(np.minimum(gt_size, pred_size)) / np.prod(np.maximum(gt_size, pred_size))
    return float(1.0 - scale_iou)


def _match_predictions(
        gt_frames: Sequence[_FrameAnnotations],
        pred_frames: Sequence[_FrameAnnotations],
        lateral_threshold: float,
        radial_scale: float,
) -> dict:
    predictions = []
    for frame_index, frame in enumerate(pred_frames):
        predictions.extend(
            (float(score), frame_index, pred_index)
            for pred_index, score in enumerate(frame.scores)
        )
    predictions.sort(key=lambda item: item[0], reverse=True)

    taken = [set() for _ in gt_frames]
    tp, fp, scores = [], [], []
    errors = {
        'trans_err': [],
        'radial_err': [],
        'lateral_err': [],
        'scale_err': [],
        'orient_err': [],
        'vel_err': [],
    }

    for score, frame_index, pred_index in predictions:
        gt_boxes = gt_frames[frame_index].boxes
        pred_box = pred_frames[frame_index].boxes[pred_index]
        available = [index for index in range(len(gt_boxes)) if index not in taken[frame_index]]
        match_index = None
        if available:
            # Each GT has its own radial axis, so evaluate candidates independently.
            equivalent_distances = np.array([
                elliptical_center_distance(gt_boxes[index, :2], pred_box[None, :2], radial_scale)[0]
                for index in available
            ])
            best = int(np.argmin(equivalent_distances))
            if equivalent_distances[best] < lateral_threshold:
                match_index = available[best]

        scores.append(score)
        if match_index is None:
            tp.append(0.0)
            fp.append(1.0)
            continue

        taken[frame_index].add(match_index)
        tp.append(1.0)
        fp.append(0.0)
        gt_box = gt_boxes[match_index]
        delta = pred_box[:2] - gt_box[:2]
        gt_range = np.linalg.norm(gt_box[:2])
        radial_unit = gt_box[:2] / gt_range if gt_range > 1e-8 else np.array([1.0, 0.0])
        lateral_unit = np.array([-radial_unit[1], radial_unit[0]])
        errors['trans_err'].append(float(np.linalg.norm(delta)))
        errors['radial_err'].append(float(abs(delta @ radial_unit)))
        errors['lateral_err'].append(float(abs(delta @ lateral_unit)))
        errors['scale_err'].append(_scale_error(gt_box[3:6], pred_box[3:6]))
        errors['orient_err'].append(_yaw_error(gt_box[6], pred_box[6]))
        if len(gt_box) >= 9 and len(pred_box) >= 9:
            velocity_delta = gt_box[7:9] - pred_box[7:9]
            errors['vel_err'].append(
                float(np.linalg.norm(velocity_delta)) if np.isfinite(velocity_delta).all() else np.nan
            )

    return {
        'tp': np.asarray(tp, dtype=np.float64),
        'fp': np.asarray(fp, dtype=np.float64),
        'scores': np.asarray(scores, dtype=np.float64),
        'errors': errors,
    }


def _average_precision(tp: np.ndarray, fp: np.ndarray, num_gt: int, config: HR4DEvalConfig) -> Tuple[float, float]:
    if num_gt == 0:
        return np.nan, np.nan
    if len(tp) == 0:
        return 0.0, 0.0

    tp_cumulative = np.cumsum(tp)
    fp_cumulative = np.cumsum(fp)
    recall = tp_cumulative / num_gt
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)
    recall_grid = np.linspace(0.0, 1.0, config.num_recall_points)
    precision_interp = np.interp(recall_grid, recall, precision, right=0.0)

    first_index = round((config.num_recall_points - 1) * config.min_recall) + 1
    clipped_precision = precision_interp[first_index:] - config.min_precision
    clipped_precision[clipped_precision < 0] = 0
    ap = float(np.clip(np.mean(clipped_precision) / (1.0 - config.min_precision), 0.0, 1.0))
    return ap, float(recall[-1])


def _mean_or_nan(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else np.nan


def evaluate_hr4d(
        gt_annos: Sequence[dict],
        pred_annos: Sequence[dict],
        class_names: Optional[Sequence[str]] = None,
        config: Optional[HR4DEvalConfig] = None,
) -> dict:
    """Evaluate predictions with the HR-4D nuScenes-derived protocol."""

    config = config or HR4DEvalConfig()
    gt_annos, pred_annos = _align_annotations_by_frame_id(gt_annos, pred_annos)

    gt_frames = _filter_region(_normalize_annotations(gt_annos, is_prediction=False, config=config), config)
    pred_frames = _filter_region(_normalize_annotations(pred_annos, is_prediction=True, config=config), config)
    if class_names is None:
        class_names = sorted({str(name) for frame in gt_frames for name in frame.names})
    class_names = tuple(class_names)

    segments = (None,) + config.distance_bins
    results = {
        'protocol': {
            'matching': (
                'nuScenes-style greedy confidence matching with GT-aligned BEV ellipses'
                if config.normalized_sensor_type == 'RADAR'
                else 'nuScenes-style greedy confidence matching with equal-weight BEV center distance'
            ),
            'sensor_type': config.normalized_sensor_type,
            'lateral_thresholds': list(config.lateral_thresholds),
            'radial_scale': config.matching_radial_scale,
            'region': {
                'forward': [0.0, config.max_forward],
                'lateral': [-config.max_lateral, config.max_lateral],
                'radial': [0.0, config.distance_bins[-1][1]],
                'fan_degrees': [-config.max_abs_angle_deg, config.max_abs_angle_deg],
            },
        },
        'class_names': list(class_names),
        'segments': {},
        'vehicle_dynamic': None,
    }

    for distance_range in segments:
        segment_key = _segment_name(distance_range)
        segment_result = {'classes': {}}
        include_upper = distance_range == config.distance_bins[-1]
        for class_name in class_names:
            gt_class = _class_segment_data(gt_frames, class_name, distance_range, include_upper)
            pred_class = _class_segment_data(pred_frames, class_name, distance_range, include_upper)
            num_gt = sum(len(frame.boxes) for frame in gt_class)
            num_pred = sum(len(frame.boxes) for frame in pred_class)
            ap_by_threshold, recall_by_threshold = {}, {}
            tp_errors = {}

            for lateral_threshold in config.lateral_thresholds:
                matched = _match_predictions(
                    gt_class,
                    pred_class,
                    lateral_threshold=lateral_threshold,
                    radial_scale=config.matching_radial_scale,
                )
                ap, recall = _average_precision(matched['tp'], matched['fp'], num_gt, config)
                threshold_key = f'{lateral_threshold:g}'
                ap_by_threshold[threshold_key] = ap
                recall_by_threshold[threshold_key] = recall
                if lateral_threshold == config.tp_lateral_threshold:
                    tp_errors = {
                        name: _mean_or_nan(values)
                        for name, values in matched['errors'].items()
                    }

            segment_result['classes'][class_name] = {
                'num_gt': num_gt,
                'num_pred': num_pred,
                'ap_by_lateral_threshold': ap_by_threshold,
                'recall_by_lateral_threshold': recall_by_threshold,
                'mean_ap': _mean_or_nan(list(ap_by_threshold.values())),
                'tp_errors': tp_errors,
            }

        segment_result['mean_ap'] = _mean_or_nan([
            class_result['mean_ap'] for class_result in segment_result['classes'].values()
        ])
        results['segments'][segment_key] = segment_result
    results['mean_ap'] = results['segments']['overall']['mean_ap']

    gt_vehicle_dynamic = _vehicle_dynamic_data(gt_frames, config, is_gt=True)
    if gt_vehicle_dynamic is not None:
        pred_vehicle_dynamic = _vehicle_dynamic_data(pred_frames, config, is_gt=False)
        assert pred_vehicle_dynamic is not None
        num_gt = sum(len(frame.boxes) for frame in gt_vehicle_dynamic)
        num_pred = sum(len(frame.boxes) for frame in pred_vehicle_dynamic)
        ap_by_threshold, recall_by_threshold = {}, {}
        for lateral_threshold in config.lateral_thresholds:
            matched = _match_predictions(
                gt_vehicle_dynamic,
                pred_vehicle_dynamic,
                lateral_threshold=lateral_threshold,
                radial_scale=config.matching_radial_scale,
            )
            ap, recall = _average_precision(matched['tp'], matched['fp'], num_gt, config)
            threshold_key = f'{lateral_threshold:g}'
            ap_by_threshold[threshold_key] = ap
            recall_by_threshold[threshold_key] = recall

        vehicle_dynamic_result = {
            'class_name': 'Vehicle',
            'distance_range': list(config.vehicle_dynamic_distance_range),
            'dynamic_source': (
                'dynamic_gt_candidate, speed_abs_xy_candidate, '
                'gt_velocity_xy_from_radar_candidate, or GT box velocity'
            ),
            'num_gt': num_gt,
            'num_pred': num_pred,
            'ap_by_lateral_threshold': ap_by_threshold,
            'recall_by_lateral_threshold': recall_by_threshold,
            'mean_ap': _mean_or_nan(list(ap_by_threshold.values())),
        }
        results['vehicle_dynamic'] = vehicle_dynamic_result
        results[f'vehicle_dynamic_{_range_key(config.vehicle_dynamic_distance_range)}'] = vehicle_dynamic_result
    return results


def format_evaluation_results(results: dict) -> Tuple[str, Dict[str, float]]:
    """Format detailed HR-4D metrics for OpenPCDet logging."""

    def add_finite_metric(metrics, key, value):
        if np.isfinite(value):
            metrics[key] = value

    def format_number(value, precision=3):
        if value is None:
            return '-'
        try:
            value = float(value)
        except (TypeError, ValueError):
            return '-'
        if not np.isfinite(value):
            return '-'
        text = f'{value:.{precision}f}'.rstrip('0').rstrip('.')
        return text if text else '0'

    def format_instances(value):
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return '-'

    def display_segment_name(segment_name):
        if segment_name != 'overall':
            return segment_name
        radial_range = results['protocol']['region']['radial']
        return f'{radial_range[0]:g}-{radial_range[1]:g}m'

    headers = ['Category', '#Instances', 'AP', 'ATE', 'ASE', 'AOE', 'AVE']
    align_right = {'#Instances', 'AP', 'ATE', 'ASE', 'AOE', 'AVE'}

    def make_table(rows, widths):
        def cell(header, value):
            value = str(value)
            if header in align_right:
                return value.rjust(widths[header])
            return value.ljust(widths[header])

        header_line = '| ' + ' | '.join(cell(header, header) for header in headers) + ' |'
        separator = []
        for header in headers:
            dash_count = max(widths[header], 3)
            if header in align_right:
                separator.append('-' * (dash_count - 1) + ':')
            else:
                separator.append(':' + '-' * (dash_count - 1))
        separator_line = '| ' + ' | '.join(separator) + ' |'
        body_lines = [
            '| ' + ' | '.join(cell(header, row[header]) for header in headers) + ' |'
            for row in rows
        ]
        return '\n'.join([header_line, separator_line, *body_lines])

    lines = [
        'HR-4D nuScenes-derived elliptical center-distance evaluation',
        'AP thresholds use lateral radius; radial radius is 2x by default.',
    ]
    flat = {}
    vehicle_dynamic = results.get('vehicle_dynamic')
    extra_tables = []
    if vehicle_dynamic is not None:
        dynamic_range = vehicle_dynamic['distance_range']
        dynamic_label = f'{dynamic_range[0]:g}-{dynamic_range[1]:g}m'
        dynamic_key = f'vehicle_dynamic_{_range_key(dynamic_range)}'
        extra_tables.append((
            f'{dynamic_label} Vehicle dynamic-only',
            [{
                'Category': f'Vehicle_dynamic_{dynamic_label}',
                '#Instances': format_instances(vehicle_dynamic['num_gt']),
                'AP': format_number(vehicle_dynamic['mean_ap']),
                'ATE': '-',
                'ASE': '-',
                'AOE': '-',
                'AVE': '-',
            }]
        ))
        add_finite_metric(flat, f'{dynamic_key}_map', vehicle_dynamic['mean_ap'])
        add_finite_metric(flat, f'hr4d/{dynamic_key}/mean_ap', vehicle_dynamic['mean_ap'])
        add_finite_metric(flat, f'hr4d/{dynamic_key}/num_gt', vehicle_dynamic['num_gt'])
        add_finite_metric(flat, f'hr4d/{dynamic_key}/num_pred', vehicle_dynamic['num_pred'])
        for threshold, ap in vehicle_dynamic['ap_by_lateral_threshold'].items():
            add_finite_metric(flat, f'hr4d/{dynamic_key}/AP@lat_{threshold}', ap)
        for threshold, recall in vehicle_dynamic['recall_by_lateral_threshold'].items():
            add_finite_metric(flat, f'hr4d/{dynamic_key}/recall@lat_{threshold}', recall)

    segment_tables = []
    for segment_name, segment in results['segments'].items():
        segment_label = display_segment_name(segment_name)
        rows = []
        for class_name in results['class_names']:
            class_result = segment['classes'][class_name]
            errors = class_result['tp_errors']
            rows.append({
                'Category': f'{class_name}_{segment_label}',
                '#Instances': format_instances(class_result['num_gt']),
                'AP': format_number(class_result['mean_ap']),
                'ATE': format_number(errors.get('trans_err')),
                'ASE': format_number(errors.get('scale_err')),
                'AOE': format_number(errors.get('orient_err')),
                'AVE': format_number(errors.get('vel_err')),
            })
            prefix = f'hr4d/{segment_name}/{class_name}'
            add_finite_metric(flat, f'{prefix}/mean_ap', class_result['mean_ap'])
            for threshold, ap in class_result['ap_by_lateral_threshold'].items():
                add_finite_metric(flat, f'{prefix}/AP@lat_{threshold}', ap)
            for threshold, recall in class_result['recall_by_lateral_threshold'].items():
                add_finite_metric(flat, f'{prefix}/recall@lat_{threshold}', recall)
            for error_name, error in class_result['tp_errors'].items():
                add_finite_metric(flat, f'{prefix}/{error_name}', error)
        summary_errors = {}
        for error_name in ('trans_err', 'scale_err', 'orient_err', 'vel_err'):
            values = [class_result['tp_errors'].get(error_name, np.nan) for class_result in segment['classes'].values()]
            summary_errors[error_name] = _mean_or_nan(values)
        rows.append({
            'Category': 'Summary',
            '#Instances': format_instances(sum(class_result['num_gt'] for class_result in segment['classes'].values())),
            'AP': format_number(segment['mean_ap']),
            'ATE': format_number(summary_errors['trans_err']),
            'ASE': format_number(summary_errors['scale_err']),
            'AOE': format_number(summary_errors['orient_err']),
            'AVE': format_number(summary_errors['vel_err']),
        })
        segment_tables.append((segment_label, rows))
        add_finite_metric(flat, f'hr4d/{segment_name}/mean_ap', segment['mean_ap'])
    all_tables = extra_tables + segment_tables
    all_rows = [row for _, rows in all_tables for row in rows]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in all_rows))
        for header in headers
    }
    for table_label, rows in all_tables:
        lines.append(f'\n[{table_label}] Evaluation metrics')
        lines.append(make_table(rows, widths))
    add_finite_metric(flat, 'hr4d/mean_ap', results['mean_ap'])
    return '\n'.join(lines), flat


def get_evaluation_results(
        gt_annos: Sequence[dict],
        pred_annos: Sequence[dict],
        class_names: Optional[Sequence[str]] = None,
        config: Optional[HR4DEvalConfig] = None,
) -> Tuple[str, Dict[str, float]]:
    """OpenPCDet-compatible result string and flat scalar metric dictionary."""

    results = evaluate_hr4d(gt_annos, pred_annos, class_names=class_names, config=config)
    return format_evaluation_results(results)
