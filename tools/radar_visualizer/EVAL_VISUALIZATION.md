# HR-4D Evaluation Visualization

This tool is WeiKang's evaluation review workflow for detection results. It
turns an OpenPCDet `result.pkl` into a static, interactive review bundle that
shows model predictions against GT, camera images, raw Radar points, ATX LiDAR,
EM4 LiDAR and image projections.

The output is for smoke and exploratory diagnosis unless the formal experiment
gate is opened.

For a team-facing usage guide and release-note summary, see
`docs/hr4d-eval-visualization-release-note.md`.

## What It Shows

- Per-frame TP / FP / FN / localization-error counts.
- Ranked problem cases across the split.
- FP reasons: duplicate, class confusion, localization/class boundary, or
  background / context.
- FN reasons: missed detection, class confusion, or localization miss.
- Camera image with Radar point projection and 3D box projection.
- Primary interactive 3D point-cloud view with Radar, ATX LiDAR, EM4 LiDAR,
  GT boxes and prediction boxes.
- PCL-style 3D viewport presets for isometric, top, front, left, fit-to-frame
  and selected-case focus views.
- Explicit 3D orbit/pan modes plus yaw, pitch, zoom and height-scale sliders
  for arbitrary viewpoint selection.
- Radar point color modes for fixed color, RCS, Doppler and absolute Doppler
  speed (`AbsV`) in 3D, BEV and image projection views.
- The frame info panel reports the selected Radar color scale as
  min/median/max when RCS, Doppler or `AbsV` is active.
- Auxiliary BEV context with Radar, ATX LiDAR, EM4 LiDAR, GT and prediction
  layers.
- Mouse wheel zoom and drag controls in BEV and 3D views.
- Layer toggles for Radar, ATX, EM4, GT, prediction and image projection.

## Color Meaning

- Green: GT matched by a prediction.
- Red-orange: FN GT.
- Gray: ignored GT outside the evaluation region.
- Cyan: TP prediction.
- Pink: FP prediction.
- Yellow: selected or localization-warning context.
- Yellow points: Radar.
- Radar RCS mode: low-to-high values are mapped from blue to yellow.
- Radar Doppler mode: negative-to-positive values are mapped from blue through
  white to red.
- Blue points: ATX LiDAR.
- Purple points: EM4 LiDAR.

## Generate A Review Bundle

Run inside the existing official container through the existing runner:

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

Outputs:

```text
output/weikang_eval_review/user_result/index.html
output/weikang_eval_review/user_result/assets/index.json
output/weikang_eval_review/user_result/assets/frame_*.json
output/weikang_eval_review/user_result/assets/frame_*.jpg
output/weikang_eval_review/user_result/eval_diff.json
output/weikang_eval_review/user_result/eval_overlay.json
```

## Open The Review

Serve the directory with any static HTTP server:

```bash
cd output/weikang_eval_review/user_result
python -m http.server 8899
```

Then open:

```text
http://127.0.0.1:8899
```

If serving from the remote machine, forward the port through SSH first.

## Suggested Review Workflow

1. Start with `FN` cases to identify systematic missed objects by distance,
   class and point density.
2. Review high-score `FP` cases to separate duplicate boxes from background
   hallucinations.
3. Use `LOC` cases to inspect center, yaw and scale errors where the detector
   found the object but placed it poorly.
4. Use the 3D point-cloud view to inspect height, pitch/yaw perception and
   vertical support around difficult boxes.
5. Use `TOP`, `FRONT`, `LEFT`, `FIT` and `FOCUS` to switch between PCL-style
   inspection viewpoints for selected cases.
6. Toggle Radar / ATX / EM4 layers to understand whether a failure is caused by
   radar sparsity, LiDAR label support, or model post-processing.
7. Check the image projection for obvious calibration, synchronization or box
   orientation problems.

## Notes

- Matching follows the HR-4D elliptical center-distance protocol.
- GT outside the HR-4D evaluation region is kept as visual context, but it is
  rendered as `IGNORE GT` and is not counted as TP/FN.
- Class names are normalized using the HR-4D class mapping, so `Car`, `Truck`
  and `Bus` are compared as `Vehicle`.
- The visualizer uses synchronized points from `tools/radar_visualizer/server.py`.
- Point clouds are display-downsampled by the existing radar visualizer data
  layer. Source and display point counts are shown in the UI.
- `eval_diff.json` contains all frames and cases; the HTML bundle only exports
  the selected highest-priority cases and their frames.
