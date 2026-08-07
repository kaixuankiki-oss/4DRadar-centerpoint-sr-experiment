import unittest

import numpy as np

from tools.radar_visualizer.eval_diff import (
    EvalDiffConfig,
    build_eval_report,
    canonical_class,
    match_frame,
    overlay_frames_from_report,
    select_review_frames,
)
from tools.radar_visualizer.export_eval_visualization import HTML_TEMPLATE


def make_info(frame_id="frame-1"):
    return {
        "frame_id": frame_id,
        "sequence_id": "seq-1",
        "timestamp": 1.0,
        "annos": {
            "names": np.array(["Car", "Pedestrian"], dtype=object),
            "boxes_3d": np.array(
                [
                    [40.0, 0.0, 0.0, 4.5, 1.9, 1.7, 0.0],
                    [30.0, 4.0, 0.0, 0.8, 0.6, 1.7, 0.0],
                ],
                dtype=np.float32,
            ),
            "gt_id": np.array(["gt-car", "gt-ped"], dtype=object),
        },
    }


class EvalVisualizationTest(unittest.TestCase):
    def test_canonical_class_uses_hr4d_mapping(self):
        self.assertEqual(canonical_class("Car"), "Vehicle")
        self.assertEqual(canonical_class("Truck"), "Vehicle")
        self.assertEqual(canonical_class("Tricycle"), "Cyclist")

    def test_match_frame_finds_tp_fp_and_fn(self):
        prediction = {
            "frame_id": "frame-1",
            "names": np.array(["Car", "Car"], dtype=object),
            "pred_boxes": np.array(
                [
                    [40.4, 0.2, 0.0, 4.5, 1.9, 1.7, 0.0],
                    [90.0, 8.0, 0.0, 4.5, 1.9, 1.7, 0.0],
                ],
                dtype=np.float32,
            ),
            "score": np.array([0.9, 0.8], dtype=np.float32),
        }
        frame = match_frame(make_info(), prediction, 0, EvalDiffConfig(score_threshold=0.1))

        self.assertEqual(frame["summary"]["tp"], 1)
        self.assertEqual(frame["summary"]["fp"], 1)
        self.assertEqual(frame["summary"]["fn"], 1)
        self.assertEqual(frame["predictions"][0]["match_status"], "tp")
        self.assertEqual(frame["predictions"][1]["match_status"], "fp")
        self.assertEqual(frame["gt"][1]["match_status"], "fn")

    def test_loc_case_is_ranked_for_large_matched_error(self):
        prediction = {
            "frame_id": "frame-1",
            "names": np.array(["Car"], dtype=object),
            "pred_boxes": np.array([[41.8, 0.0, 0.0, 4.5, 1.9, 1.7, 0.0]], dtype=np.float32),
            "score": np.array([0.95], dtype=np.float32),
        }
        config = EvalDiffConfig(match_lateral_threshold=2.0, loc_warning_threshold=0.5)
        frame = match_frame(make_info(), prediction, 0, config)

        self.assertEqual(frame["summary"]["tp"], 1)
        self.assertEqual(frame["summary"]["loc"], 1)
        self.assertEqual(frame["cases"][0]["type"], "FN")
        self.assertTrue(any(case["type"] == "LOC" for case in frame["cases"]))

    def test_out_of_region_gt_is_kept_as_ignore_context(self):
        info = make_info()
        info["annos"]["names"] = np.array(["Car"], dtype=object)
        info["annos"]["boxes_3d"] = np.array([[40.0, 30.0, 0.0, 4.5, 1.9, 1.7, 0.0]], dtype=np.float32)
        info["annos"]["gt_id"] = np.array(["gt-ignore"], dtype=object)
        prediction = {
            "frame_id": "frame-1",
            "names": np.array([], dtype=object),
            "pred_boxes": np.empty((0, 7), dtype=np.float32),
            "score": np.array([], dtype=np.float32),
        }

        frame = match_frame(info, prediction, 0, EvalDiffConfig(score_threshold=0.1))

        self.assertEqual(frame["summary"]["gt"], 0)
        self.assertEqual(frame["summary"]["fn"], 0)
        self.assertFalse(frame["gt"][0]["in_region"])
        self.assertNotIn("match_status", frame["gt"][0])

    def test_html_template_contains_interactive_3d_view(self):
        self.assertIn('id="view3dCanvas"', HTML_TEMPLATE)
        self.assertIn("view3d-panel", HTML_TEMPLATE)
        self.assertIn("Primary 3D Point Cloud", HTML_TEMPLATE)
        self.assertIn("Aux BEV Context", HTML_TEMPLATE)
        self.assertIn("function render3d()", HTML_TEMPLATE)
        self.assertIn("function bind3dControls()", HTML_TEMPLATE)
        self.assertIn('data-view3d="top"', HTML_TEMPLATE)
        self.assertIn("function fit3dToFrame()", HTML_TEMPLATE)
        self.assertIn("function focus3dOnSelected", HTML_TEMPLATE)
        self.assertIn("function bind3dControlInputs()", HTML_TEMPLATE)
        self.assertIn('id="view3dYaw"', HTML_TEMPLATE)
        self.assertIn('data-view3d-mode="pan"', HTML_TEMPLATE)
        self.assertIn('id="view3dZScale"', HTML_TEMPLATE)
        self.assertIn('id="radarColorMode"', HTML_TEMPLATE)
        self.assertIn("function radarPointColor", HTML_TEMPLATE)
        self.assertIn("function radarColorStatsLabel", HTML_TEMPLATE)
        self.assertIn("Radar Doppler", HTML_TEMPLATE)
        self.assertIn("Radar RCS", HTML_TEMPLATE)
        self.assertIn('id="focus3dView"', HTML_TEMPLATE)
        self.assertIn("Reset 3D", HTML_TEMPLATE)
        self.assertIn("touch-action: none", HTML_TEMPLATE)

    def test_report_selection_and_overlay_generation(self):
        infos = [make_info("frame-1")]
        predictions = [
            {
                "frame_id": "frame-1",
                "names": np.array(["Car"], dtype=object),
                "pred_boxes": np.array([[40.0, 0.0, 0.0, 4.5, 1.9, 1.7, 0.0]], dtype=np.float32),
                "score": np.array([0.9], dtype=np.float32),
            }
        ]
        report = build_eval_report(infos, predictions, EvalDiffConfig(max_cases=10, max_frames=10))
        cases, frame_indices = select_review_frames(report, max_cases=10, max_frames=10)
        overlay = overlay_frames_from_report(report)

        self.assertEqual(frame_indices, [0])
        self.assertGreaterEqual(len(cases), 1)
        self.assertIn("frame-1", overlay["frames"])
        self.assertEqual(overlay["frames"]["frame-1"][0]["match_status"], "tp")
        self.assertIn("eval_id", overlay["frames"]["frame-1"][0])


if __name__ == "__main__":
    unittest.main()
