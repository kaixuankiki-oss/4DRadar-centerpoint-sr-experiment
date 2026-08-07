# L4 CenterPoint + PointPillar red-zone baseline

Date: 2026-06-15

## Added config

- Branch/worktree: `HR-4D`
- Main config: `tools/cfgs/radar_models/l4/centerpoint_pointpillar_redzone_baseline_l4.yaml`
- Smoke config used only for the 10 epoch run:
  `tools/cfgs/radar_models/l4/centerpoint_pointpillar_redzone_baseline_l4_10epoch_smoke.yaml`

## Design

- Dataset protocol follows the previous HR-4D experiments:
  `data/1000_original_data` with split `splits/hr4d_1000_v1`
  (`800` train samples, `200` test samples).
- The config is self-contained under `DATA_CONFIG` instead of relying on a
  nested dataset `_BASE_CONFIG_`, because the current HR-4D config parser only
  expands one base level for this entry point.
- Stacked frames use L4 `RadarDataset.MAX_SWEEPS=3`, matching the red-zone
  baseline `SEQUENCE_CONFIG.NUM_FRAMES=3`.
- Detection range matches the red-zone baseline:
  `[0.0, -20.0, -3.0, 200.0, 20.0, 4.0]`.
- Pillar settings match the red-zone baseline:
  `VOXEL_SIZE=[0.2, 0.2, 7.0]`,
  `MAX_POINTS_PER_VOXEL=32`,
  `MAX_NUMBER_OF_VOXELS=8000/8000`.
- Model is `CenterPoint + PillarVFE + PointPillarScatter + BaseBEVBackbone + CenterHead`.
  Per requirement, feature extraction stays PointPillar-style and does not use
  red-zone `RadarPillarVFE` or `PillarAttention`.
- Input point features are
  `['x', 'y', 'z', 'Vx', 'Vy', 'RCS', 'timestamp']`.
  The L4 `RadarDataset` generates `timestamp` as the current-frame to sweep
  time difference, which is the practical equivalent of red-zone `time_diff`.
- Data augmentation only uses `random_world_scaling` with
  `WORLD_SCALE_RANGE=[0.95, 1.05]`.
  No flip, rotation, translation, GT sampling, or other augmentation is enabled.
- Main training schedule follows the red-zone baseline where compatible:
  `NUM_EPOCHS=40`, `LR=0.003`, `PCT_START=0.3`, `WARMUP_EPOCH=2`.
  The L4 entry uses `BATCH_SIZE_PER_GPU=2`.
- Evaluator is `hr4d`, because the current HR-4D `RadarDataset` supports
  `hr4d` and `kitti`; it does not implement the red-zone `hirain` evaluator.
- `DENSE_HEAD.IOU_REG_LOSS` is set to `False`. The HR-4D CenterHead path
  crashed with `UnboundLocalError: batch_box_preds_for_iou` when the red-zone
  DIoU setting was copied without an IoU prediction head. Disabling this option
  makes the config compatible with the current HR-4D CenterHead implementation.

## Dependency audit

Both YAML entry points were parsed in the HR-4D container:

- Main baseline base:
  `cfgs/path_configs/hr4d_paths.yaml`
- 10 epoch smoke base:
  `cfgs/radar_models/l4/centerpoint_pointpillar_redzone_baseline_l4.yaml`

Required fields were present after parsing:

- `CLASS_NAMES=['Car', 'LargeVehicle', 'Cyclist', 'Pedestrian']`
- `DATA_CONFIG.DATASET=RadarDataset`
- `DATA_CONFIG.DATA_PATH=data/1000_original_data`
- `DATA_CONFIG.INFO_PATH` points to `hr4d_1000_v1`
- `DATA_CONFIG.MAX_SWEEPS=3`
- `DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST` contains only
  `random_world_scaling [0.95, 1.05]`
- `MODEL.NAME=CenterPoint`
- `MODEL.VFE.NAME=PillarVFE`
- `MODEL.DENSE_HEAD.NAME=CenterHead`
- `MODEL.DENSE_HEAD.IOU_REG_LOSS=False`

Runtime data dependency in the HR-4D worktree:

```text
/workspace/4DRADAR/data/1000_original_data -> /data/1000_original_data
```

## Smoke and training validation

Single-batch forward/backward smoke passed in the HR-4D container:

```text
Loading Radar dataset.
Total samples for Radar dataset: 800
model=CenterPoint
vfe=PillarVFE
dense_head=CenterHead
num_point_features=7
loss=137.5186309814453
tb_keys=['hm_loss_head_0', 'loc_loss_head_0', 'loss_rpn', 'rpn_loss']
```

The 10 epoch training run completed and automatically evaluated epoch 10.

Command:

```bash
python train.py \
  --cfg_file cfgs/radar_models/l4/centerpoint_pointpillar_redzone_baseline_l4_10epoch_smoke.yaml \
  --extra_tag centerpoint_pointpillar_redzone_l4_10epoch_smoke \
  --workers 0 \
  --sync_bn
```

Log:

```text
/experiments/hr4d-centerpoint-pointpillar-l4-smoke-20260615/train.log
```

Checkpoint:

```text
/workspace/4DRADAR/output/radar_models/l4/centerpoint_pointpillar_redzone_baseline_l4_10epoch_smoke/centerpoint_pointpillar_redzone_l4_10epoch_smoke/ckpt/checkpoint_epoch_10.pth
```

Training loss moved down normally:

| epoch | final averaged loss |
| --- | ---: |
| 1 | about 6.68 |
| 2 | about 5.51 |
| 3 | about 5.83 |
| 4 | about 5.36 |
| 5 | about 5.04 |
| 6 | about 4.77 |
| 7 | about 4.57 |
| 8 | about 4.37 |
| 9 | about 4.28 |
| 10 | about 4.16 |

Epoch 10 HR-4D test split evaluation:

```text
recall_rcnn_0.3: 0.186317
recall_rcnn_0.5: 0.117040
recall_rcnn_0.7: 0.028399
Average predicted number of objects(200 samples): 8.580

[overall] mAP: 0.0651
Car: mAP=0.2540, GT=1134, pred=1517
LargeVehicle: mAP=0.0065, GT=174, pred=57
Cyclist: mAP=0.0000, GT=79, pred=93
Pedestrian: mAP=0.0000, GT=14, pred=6
```

The `gpustat: not found` messages in the log are non-fatal logging warnings and
did not interrupt training or evaluation.

## Point-cloud range assessment

The YAML was not changed after this assessment. The current range is kept as the
red-zone-aligned baseline range, not as a claim that it is the best coverage
range for HR-4D.

Current range:

```text
[0.0, -20.0, -3.0, 200.0, 20.0, 4.0]
```

With `VOXEL_SIZE=[0.2, 0.2, 7.0]`, the BEV grid is:

```text
[1000, 200, 1]
```

So the z axis has one pillar layer. In this PointPillar-style setup, the z range
mainly decides which points and GT centers are filtered before pillarization; it
does not add z-resolution inside the BEV feature map.

Coverage under the current range:

| split | stacked points in ROI | target GT centers in ROI | GT centers in x | GT centers in y | GT centers in z |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 49.38% | 61.04% | 85.82% | 73.06% | 96.09% |
| test | 50.63% | 63.64% | 92.13% | 70.22% | 98.84% |

Per-class GT center coverage under the current range:

| split | Car | LargeVehicle | Cyclist | Pedestrian |
| --- | ---: | ---: | ---: | ---: |
| train | 62.22% | 54.93% | 62.13% | 54.47% |
| test | 65.39% | 51.60% | 68.70% | 82.35% |

Distribution checks:

- Train point z percentiles: p5 `-6.453`, p50 `0.220`, p95 `11.173`.
- Test point z percentiles: p5 `-6.254`, p50 `0.163`, p95 `8.736`.
- Train GT center z percentiles: p1 `-2.298`, p50 `0.739`, p99 `6.451`.
- Test GT center z percentiles: p1 `-2.507`, p50 `0.718`, p99 `2.766`.
- No sample exceeded the `8000` unique-pillar cap after current range masking.
  Train p99 unique pillars/sample was about `6505`; test p99 was about `5996`.

Range comparison, keeping all other config choices unchanged:

| candidate range | train point in ROI | train GT in ROI | test point in ROI | test GT in ROI |
| --- | ---: | ---: | ---: | ---: |
| current `[0,-20,-3,200,20,4]` | 49.38% | 61.04% | 50.63% | 63.64% |
| z only `[-5,5]` | 56.05% | 61.48% | 56.70% | 63.64% |
| z only `[-8,8]` | 62.38% | 61.84% | 62.63% | 63.64% |
| y only `[-30,30]` | 55.77% | 74.23% | 59.01% | 80.03% |
| x only `[0,250]` | 50.74% | 67.64% | 51.70% | 66.35% |
| x/y `[0,250],[-30,30]` | 57.46% | 81.58% | 60.41% | 83.26% |
| wide reference `[0,-50,-5,300,50,5]` | 72.08% | 93.77% | 75.33% | 95.01% |

Conclusion:

- The z range `[-3,4]` is somewhat tight for raw stacked radar points: it keeps
  about `65%` to `68%` of points when checked on z alone.
- It is not the main reason for low target coverage, because GT center z
  coverage is already high: `96.09%` on train and `98.84%` on test.
- The dominant GT coverage bottleneck is the lateral y range. Expanding only y
  from `[-20,20]` to `[-30,30]` raises test GT center coverage from `63.64%`
  to `80.03%`.
- Expanding x from `200m` to `250m` helps less than y by itself, but x/y
  together raises test GT coverage to `83.26%`.
- For this pushed baseline, keeping the red-zone range is reasonable because
  the request was to match the red-zone baseline where possible. For a
  performance-oriented HR-4D follow-up, the next controlled experiment should
  test a wider y range and optionally a wider x range; z widening is secondary
  and should be tested mainly for point-retention effects rather than GT
  center coverage.

## Notes

- This is not a strict red-zone RadarPillar reproduction, because it does not
  use `RadarPillarVFE`, `PillarAttention`, or the `hirain` evaluator.
- It is the requested L4 baseline that keeps CenterPoint and red-zone geometry
  settings while using PointPillar/PillarVFE feature extraction and the previous
  HR-4D dataset protocol.
