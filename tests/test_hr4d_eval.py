import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / 'pcdet'
    / 'datasets'
    / 'hr4d'
    / 'hr4d_eval'
    / 'evaluation.py'
)
SPEC = importlib.util.spec_from_file_location('hr4d_evaluation', MODULE_PATH)
EVAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVAL
SPEC.loader.exec_module(EVAL)


def box(x, y, yaw=0.0):
    return [x, y, 0.0, 4.0, 2.0, 1.5, yaw]


def gt(names, boxes):
    return {'names': np.asarray(names, dtype=object), 'boxes_3d': np.asarray(boxes, dtype=float)}


def pred(names, boxes, scores):
    return {
        'name': np.asarray(names, dtype=object),
        'boxes_lidar': np.asarray(boxes, dtype=float),
        'score': np.asarray(scores, dtype=float),
    }


class TestHR4DEvaluation(unittest.TestCase):
    def test_ellipse_is_radial_gt_aligned(self):
        gt_xy = np.array([50.0, 50.0])
        radial = np.array([1.0, 1.0]) / np.sqrt(2.0)
        lateral = np.array([-1.0, 1.0]) / np.sqrt(2.0)

        radial_distance = EVAL.elliptical_center_distance(gt_xy, gt_xy + radial * 3.9)[0]
        lateral_distance = EVAL.elliptical_center_distance(gt_xy, gt_xy + lateral * 2.1)[0]

        self.assertLess(radial_distance, 2.0)
        self.assertGreater(lateral_distance, 2.0)

    def test_region_is_rectangle_intersected_with_fan(self):
        config = EVAL.HR4DEvalConfig()
        boxes = np.asarray([
            box(100.0, 0.0),
            box(-1.0, 0.0),
            box(201.0, 0.0),
            box(100.0, 21.0),
            box(10.0, 9.0),
            box(200.0, 20.0),
        ])

        mask = EVAL.evaluation_region_mask(boxes, config)

        np.testing.assert_array_equal(mask, [True, False, False, False, False, False])

    def test_perfect_predictions_are_reported_in_each_distance_bin(self):
        boxes = [box(25.0, 0.0), box(75.0, 0.0), box(125.0, 0.0), box(175.0, 0.0)]
        result = EVAL.evaluate_hr4d(
            [gt(['Car'] * 4, boxes)],
            [pred(['Car'] * 4, boxes, [0.9, 0.8, 0.7, 0.6])],
            class_names=['Car'],
        )

        self.assertAlmostEqual(result['mean_ap'], 1.0)
        for segment in ('0-50m', '50-100m', '100-150m', '150-200m'):
            self.assertEqual(result['segments'][segment]['classes']['Car']['num_gt'], 1)
            self.assertAlmostEqual(result['segments'][segment]['mean_ap'], 1.0)

    def test_radial_error_gets_twice_lateral_tolerance(self):
        config = EVAL.HR4DEvalConfig(lateral_thresholds=(2.0,), tp_lateral_threshold=2.0)
        gt_annos = [gt(['Car'], [box(100.0, 0.0)])]
        radial_result = EVAL.evaluate_hr4d(
            gt_annos,
            [pred(['Car'], [box(103.9, 0.0)], [0.9])],
            class_names=['Car'],
            config=config,
        )
        lateral_result = EVAL.evaluate_hr4d(
            gt_annos,
            [pred(['Car'], [box(100.0, 2.1)], [0.9])],
            class_names=['Car'],
            config=config,
        )

        self.assertAlmostEqual(radial_result['mean_ap'], 1.0)
        self.assertAlmostEqual(lateral_result['mean_ap'], 0.0)

    def test_matching_is_one_to_one_and_confidence_ordered(self):
        config = EVAL.HR4DEvalConfig(lateral_thresholds=(2.0,), tp_lateral_threshold=2.0)
        result = EVAL.evaluate_hr4d(
            [gt(['Car'], [box(50.0, 0.0)])],
            [pred(['Car', 'Car'], [box(50.0, 10.0), box(50.0, 0.0)], [0.99, 0.5])],
            class_names=['Car'],
            config=config,
        )
        car = result['segments']['overall']['classes']['Car']

        self.assertAlmostEqual(car['recall_by_lateral_threshold']['2'], 1.0)
        self.assertLess(car['mean_ap'], 1.0)

    def test_separate_velocity_fields_are_supported(self):
        gt_anno = gt(['Car'], [box(50.0, 0.0)])
        gt_anno['vels'] = np.asarray([[3.0, 1.0]])
        pred_anno = pred(['Car'], [box(50.0, 0.0)], [0.9])
        pred_anno['vels'] = np.asarray([[2.0, 1.0]])

        result = EVAL.evaluate_hr4d([gt_anno], [pred_anno], class_names=['Car'])

        velocity_error = result['segments']['overall']['classes']['Car']['tp_errors']['vel_err']
        self.assertAlmostEqual(velocity_error, 1.0)

    def test_custom_last_distance_bin_includes_its_upper_bound(self):
        config = EVAL.HR4DEvalConfig(
            distance_bins=((0.0, 10.0), (10.0, 20.0)),
            max_forward=20.0,
        )
        result = EVAL.evaluate_hr4d(
            [gt(['Car'], [box(20.0, 0.0)])],
            [pred(['Car'], [box(20.0, 0.0)], [0.9])],
            class_names=['Car'],
            config=config,
        )

        self.assertEqual(result['segments']['10-20m']['classes']['Car']['num_gt'], 1)

    def test_eval_range_builds_50m_bins_to_300m(self):
        config = EVAL.HR4DEvalConfig.from_ranges(eval_range=[0, 300])
        result = EVAL.evaluate_hr4d(
            [gt(['Car', 'Car'], [box(225.0, 0.0), box(275.0, 0.0)])],
            [pred(['Car', 'Car'], [box(225.0, 0.0), box(275.0, 0.0)], [0.9, 0.8])],
            class_names=['Car'],
            config=config,
        )

        self.assertIn('200-250m', result['segments'])
        self.assertIn('250-300m', result['segments'])
        self.assertEqual(result['segments']['200-250m']['classes']['Car']['num_gt'], 1)
        self.assertEqual(result['segments']['250-300m']['classes']['Car']['num_gt'], 1)

    def test_openpcdet_formatter_returns_scalar_metrics(self):
        result_str, metrics = EVAL.get_evaluation_results(
            [gt(['Car'], [box(20.0, 0.0)])],
            [pred(['Car'], [box(20.0, 0.0)], [0.9])],
            class_names=['Car'],
        )

        self.assertIn('[0-200m] Evaluation metrics', result_str)
        self.assertAlmostEqual(metrics['hr4d/mean_ap'], 1.0)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_formatter_omits_metrics_for_classes_without_gt(self):
        _, metrics = EVAL.get_evaluation_results(
            [gt(['Car'], [box(20.0, 0.0)])],
            [pred(['Car'], [box(20.0, 0.0)], [0.9])],
            class_names=['Car', 'Cyclist'],
        )

        self.assertFalse(any('/Cyclist/' in key for key in metrics))

    def test_vehicle_dynamic_100_200_metric_merges_vehicle_classes(self):
        gt_anno = gt(
            ['Car', 'LargeV', 'Car', 'Pedestrian', 'Car'],
            [
                box(120.0, 0.0),
                box(180.0, 0.0),
                box(80.0, 0.0),
                box(130.0, 0.0),
                box(160.0, 0.0),
            ],
        )
        gt_anno['dynamic_gt_candidate'] = np.asarray([True, True, True, True, False])
        pred_anno = pred(
            ['Car', 'LargeV', 'Car', 'Pedestrian', 'Car'],
            [
                box(120.0, 0.0),
                box(180.0, 0.0),
                box(80.0, 0.0),
                box(130.0, 0.0),
                box(160.0, 0.0),
            ],
            [0.99, 0.98, 0.97, 0.96, 0.95],
        )

        result_str, metrics = EVAL.get_evaluation_results(
            [gt_anno],
            [pred_anno],
            class_names=['Car', 'LargeV', 'Pedestrian'],
        )

        self.assertIn('[100-200m Vehicle dynamic-only] Evaluation metrics', result_str)
        self.assertGreater(metrics['vehicle_dynamic_100_200_map'], 0.99)
        self.assertLess(metrics['vehicle_dynamic_100_200_map'], 1.0)
        self.assertAlmostEqual(
            metrics['hr4d/vehicle_dynamic_100_200/mean_ap'],
            metrics['vehicle_dynamic_100_200_map'],
        )
        self.assertEqual(metrics['hr4d/vehicle_dynamic_100_200/num_gt'], 2)
        self.assertEqual(metrics['hr4d/vehicle_dynamic_100_200/num_pred'], 3)

    def test_vehicle_dynamic_metric_is_omitted_without_dynamic_gt_fields(self):
        _, metrics = EVAL.get_evaluation_results(
            [gt(['Car'], [box(120.0, 0.0)])],
            [pred(['Car'], [box(120.0, 0.0)], [0.9])],
            class_names=['Car'],
        )

        self.assertNotIn('vehicle_dynamic_100_200_map', metrics)

    def test_vehicle_dynamic_metric_uses_configured_range_key(self):
        gt_anno = gt(['Car', 'Car'], [box(120.0, 0.0), box(250.0, 0.0)])
        gt_anno['dynamic_gt_candidate'] = np.asarray([True, True])
        config = EVAL.HR4DEvalConfig.from_ranges(
            eval_range=[0, 300],
            far_dynamic_vehicle=[150, 300],
        )

        result_str, metrics = EVAL.get_evaluation_results(
            [gt_anno],
            [pred(['Car', 'Car'], [box(120.0, 0.0), box(250.0, 0.0)], [0.9, 0.8])],
            class_names=['Car'],
            config=config,
        )

        self.assertIn('[150-300m Vehicle dynamic-only] Evaluation metrics', result_str)
        self.assertIn('vehicle_dynamic_150_300_map', metrics)
        self.assertEqual(metrics['hr4d/vehicle_dynamic_150_300/num_gt'], 1)
        self.assertNotIn('vehicle_dynamic_100_200_map', metrics)


if __name__ == '__main__':
    unittest.main()
