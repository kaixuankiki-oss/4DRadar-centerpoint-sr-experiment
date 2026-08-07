# HR-4D Offline Tracking Smoke Tool

This folder contains WeiKang's lightweight offline DET -> TRACK post-process.
It is intended for smoke and exploratory visualization before the formal
experiment gate opens.

## Inputs

- `--infos`: HR-4D frame order and sequence metadata.
- `--predictions`: OpenPCDet `result.pkl` with `frame_id`, `boxes_3d` or
  `boxes_lidar`, `score` or `pred_scores`, and `name` or `pred_labels`.
- `--use-gt-as-detections`: uses GT boxes as detections only to validate the
  offline tracking and video chain.

The tracker resets whenever `sequence_id` changes, matching the HR-4D split
contract. It also resets on large timestamp gaps by default
(`--max-time-gap 2.0`) because split PKLs can be sparse rather than continuous
video streams.

## Quick Smoke

Run inside the existing official container through the existing runner:

```bash
/usr/local/bin/hr4d-run python tools/tracking/hr4d_offline_tracker.py \
  --infos data/1000_original_data/splits/hr4d_1000_v1/infos_test_200.pkl \
  --use-gt-as-detections \
  --max-frames 40 \
  --draw-detections \
  --output-video output/weikang_tracking/gt_tracks_bev_smoke.mp4
```

## Model Result

```bash
/usr/local/bin/hr4d-run python tools/tracking/hr4d_offline_tracker.py \
  --infos data/1000_original_data/splits/hr4d_1000_v1/infos_test_200.pkl \
  --predictions output/path/to/result.pkl \
  --score-threshold 0.15 \
  --distance-threshold 6.0 \
  --max-time-gap 2.0 \
  --max-age 3 \
  --draw-detections \
  --output-video output/weikang_tracking/model_tracks_bev.mp4
```

## Outputs

- `hr4d_tracks_overlay.json`: frame-level detections, tracks, IDs, velocity,
  and short history. This is the intended bridge to the radar visualizer.
- `hr4d_tracks.pkl`: OpenPCDet-style tracked predictions with `track_id`.
- Optional MP4: BEV demo video for qualitative stability review.

Treat these outputs as smoke/exploratory evidence unless the manager opens the
formal experiment entrance.
