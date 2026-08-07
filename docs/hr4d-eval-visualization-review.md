# HR-4D Evaluation Visualization Review Guide

This document records the current evaluation visualization workflow for the
WeiKang post-processing and visualization role. It complements
`tools/radar_visualizer/EVAL_VISUALIZATION.md`, which is the command reference.
For a team-facing usage and release-note summary, see
`docs/hr4d-eval-visualization-release-note.md`.

## Goal

The review tool is built to answer one question quickly:

> Where does the detector disagree with GT, and what evidence explains the
> disagreement?

A useful evaluation visualizer should not only replay predictions. It should
rank failure cases, expose FP/FN/localization patterns, and put every case next
to the raw sensor evidence: camera image, Radar point cloud, ATX LiDAR, EM4
LiDAR, GT boxes and prediction boxes.

## Implemented Capabilities

- Converts OpenPCDet `result.pkl` into a static review bundle.
- Computes frame-level `TP / FP / FN / LOC` counts using HR-4D elliptical
  center-distance matching.
- Ranks high-priority problem cases across the split.
- Classifies FP/FN context:
  - `missed_detection`
  - `class_confusion`
  - `duplicate`
  - `localization_miss`
  - `background_or_out_of_roi_context`
- Shows camera image with Radar projection points.
- Projects GT and prediction 3D boxes onto the camera image.
- Shows an interactive 3D view with Radar, ATX LiDAR, EM4 LiDAR, GT and
  prediction layers.
- Supports PCL-style 3D view presets, orbit, pan, zoom, height scaling and
  selected-case focus.
- Shows an auxiliary BEV with Radar, ATX LiDAR, EM4 LiDAR, GT and prediction
  layers.
- Supports BEV mouse-wheel zoom, drag pan, range presets and layer toggles.
- Supports Radar `RCS`, `Doppler` and `AbsV` point coloring in 3D, BEV and
  image projection views.
- Exports all detailed matching data to `eval_diff.json`.

## Main Files

```text
tools/radar_visualizer/eval_diff.py
tools/radar_visualizer/export_eval_visualization.py
tools/radar_visualizer/EVAL_VISUALIZATION.md
tests/test_eval_visualization.py
```

The existing synchronized visualizer data layer is reused:

```text
tools/radar_visualizer/server.py
```

## Current Example Result

Input:

```text
infos:       data/1000_original_data/splits/hr4d_1000_v1/infos_test_200.pkl
prediction:  output/weikang_tracking/user_result.pkl
data root:   data/1000_original_data
```

Generated bundle:

```text
output/weikang_eval_review/user_result/index.html
output/weikang_eval_review/user_result/assets/index.json
output/weikang_eval_review/user_result/eval_diff.json
output/weikang_eval_review/user_result/eval_overlay.json
```

Summary for this example:

| Metric | Value |
| --- | ---: |
| Frames | 200 |
| GT boxes in eval region | 1409 |
| Predictions in eval region | 1437 |
| TP | 756 |
| FP | 681 |
| FN | 653 |
| LOC warnings | 31 |
| Selected review cases | 61 |
| Selected review frames | 40 |

Highest-priority case in this run:

```text
type: FN
reason: missed_detection
class: Vehicle
frame_index: 108
frame_id: fc0e3bb9-9325-4c6a-b52e-d1eeaaf6e1e0
gt_id: 32e80a8b-8bd0-4128-a94c-72d3f8c516fa
nearest_pred_index: 7
nearest_pred_score: 0.2225
nearest_distance: 88.4701
```

## Visual Encoding

| Element | Color / Meaning |
| --- | --- |
| GT matched by prediction | Green |
| FN GT | Red-orange |
| Ignored GT outside evaluation region | Gray |
| TP prediction | Cyan |
| FP prediction | Pink |
| Selected / localization-warning context | Yellow |
| Radar points | Yellow |
| ATX LiDAR points | Blue |
| EM4 LiDAR points | Purple |

## Recommended Review Workflow

1. Start with `FN` cases to inspect missed vehicles or pedestrians by distance,
   class and point density.
2. Review high-score `FP` cases to separate duplicate predictions from
   background hallucinations.
3. Inspect `LOC` cases where the detector found the object but has large center,
   yaw or scale error.
4. Use the primary 3D point-cloud view to inspect object height, vertical
   sparsity and box orientation when BEV alone is ambiguous.
5. Use `ISO`, `TOP`, `FRONT`, `LEFT`, `FIT` and `FOCUS` to reproduce
   PCL-style viewpoint changes around difficult cases.
6. Adjust the 3D `Yaw`, `Pitch`, `Zoom` and `Z` controls when a preset is not
   enough to inspect a failure from the needed angle.
7. Switch Radar color between fixed, `RCS`, `Doppler` and `AbsV` to inspect
   reflectivity and radial velocity support around GT/prediction boxes; use the
   frame info min/median/max scale to interpret the colors.
8. Toggle Radar / ATX / EM4 layers to distinguish model behavior from data
   sparsity or label support.
9. Check image projection for calibration, synchronization or orientation
   anomalies.
10. Record representative frame IDs and case IDs before reporting a qualitative
   finding.

## Run Command

```bash
/usr/local/bin/hr4d-run python tools/radar_visualizer/export_eval_visualization.py \
  --infos data/1000_original_data/splits/hr4d_1000_v1/infos_test_200.pkl \
  --data-root data/1000_original_data \
  --predictions output/weikang_tracking/user_result.pkl \
  --score-threshold 0.15 \
  --match-lateral-threshold 2.0 \
  --loc-warning-threshold 1.0 \
  --max-cases 80 \
  --max-frames 40 \
  --output-dir output/weikang_eval_review/user_result
```

Open the bundle with a static server:

```bash
cd output/weikang_eval_review/user_result
python -m http.server 8899
```

Then open:

```text
http://127.0.0.1:8899
```

## Verification

The implementation was checked with:

```bash
python -m unittest -q \
  tests.test_package_version \
  tests.test_hr4d_eval \
  tests.test_radar_ego_doppler \
  tests.test_radar_pillar_vfe \
  tests.test_eval_visualization
```

Expected result:

```text
Ran 21 tests
OK
```

The generated HTML was also opened in the in-app browser and manually checked
for:

- case list loading,
- `ALL / FN / FP / LOC` filters,
- case click behavior,
- camera image and projection rendering,
- interactive 3D and BEV rendering,
- Radar / ATX / EM4 layer toggles,
- Radar `RCS / Doppler / AbsV` color modes,
- ignored GT gray rendering.

## Current Limitations

- Static review bundles can be large because selected frames include dense
  point-cloud JSON for Radar, ATX and EM4.
- The bundle exports selected high-priority frames, while `eval_diff.json`
  stores the full split-level details.
- Tracking metrics such as ID switch and fragmentation are not yet included in
  this detection review page.
- The browser review is intended for exploratory diagnosis, not a formal metric
  replacement.
