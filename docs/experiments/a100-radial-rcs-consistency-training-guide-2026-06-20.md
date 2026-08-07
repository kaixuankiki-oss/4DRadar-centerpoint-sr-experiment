# A100 radial/RCS consistency training guide

This branch prepares the FRONT-only follow-up to exp20-24.

## Active scope

- Branch: `codex/a100-radial-rcs-consistency-20260620`
- Worktree: `/workspace/worktrees/a100_radial_rcs_consistency_20260620`
- Session root: `/experiments/a100_radial_rcs_consistency_20260620`
- Output root: `/experiments/a100_radial_rcs_consistency_output`
- Rebased worktree for HR-4D eval update: `/workspace/worktrees/a100_radial_rcs_consistency_rebase_20260622`
- Target remote branch after rebase: `origin/codex/a100-radial-rcs-consistency-20260620` based on latest `origin/HR-4D`
- Temporary A100 rebase branch/worktree: `codex/a100-radial-rcs-consistency-rebase-20260622` / `/workspace/worktrees/a100_radial_rcs_consistency_rebase_20260622`
- Dataset: `/data/highway_sweeps/highway_train.pkl`, `/data/highway_sweeps/highway_test.pkl`
- Sensor: `RADAR_FRONT` only
- Cancelled for this suite: `RADAR_FRONT_TOP`, dual-front, and multi-radar fusion

## Frozen protocol

- `POINT_CLOUD_RANGE=[0,-20,-8,200,20,8]`
- `POST_CENTER_LIMIT_RANGE=[0,-20,-8,200,20,8]`
- `MAX_SWEEPS=5`
- `MIN_GT_POINTS=3`
- Evaluator: fixed HR-4D evaluator
- Training: 40 epochs from scratch

## First batch

1. `exp25_radial_gate_low_pred`
2. `exp26_radial_gate_wider_hidden`
3. `exp27_radial_rcs_consistency_gate`
4. `exp29` offline threshold/NMS analysis only

`exp28_rcs_only_temporal_consistency` is prepared as an optional isolation experiment, not part of the first required queue unless manager approves.

## Reporting requirement

Every successful result report and Feishu notification must include a baseline table with:

- historical A100 single-frame baseline
- exp20 naive 5-sweep
- exp23 radial consistency
- current experiment

Required metrics are `pred/frame`, `overall`, `0-50m`, `50-100m`, `100-150m`, `150-200m`, `Car`, `LargeVehicle`, `Cyclist`, and `Pedestrian`. `LargeVehicle` is parsed with `LargeVehicle` -> `LargeV` alias fallback.

## Validation commands

```bash
cd /workspace/worktrees/a100_radial_rcs_consistency_20260620
python tools/experiments/validate_radial_wing_configs.py \
  tools/cfgs/radar_models/exp25_radial_gate_low_pred.yaml \
  tools/cfgs/radar_models/exp26_radial_gate_wider_hidden.yaml \
  tools/cfgs/radar_models/exp27_radial_rcs_consistency_gate.yaml
```

```bash
cd /workspace/worktrees/a100_radial_rcs_consistency_rebase_20260622/tools
CUDA_VISIBLE_DEVICES=4 python experiments/smoke_one_batch_radial_wing.py \
  --cfg_file cfgs/radar_models/exp25_radial_gate_low_pred.yaml \
  --batch_size 1 --workers 1
```

Repeat the smoke command for exp26 and exp27.

## Formal training command

Do not launch this until manager explicitly approves after smoke validation. For the HR-4D evaluator-update rebase, run from `/workspace/worktrees/a100_radial_rcs_consistency_rebase_20260622` or from the pushed rebased branch, not from the old pre-rebase worktree.

```bash
cd /workspace/worktrees/a100_radial_rcs_consistency_rebase_20260622
python tools/experiments/a100_radial_rcs_supervisor.py \
  --repo /workspace/worktrees/a100_radial_rcs_consistency_rebase_20260622 \
  --root /experiments/a100_radial_rcs_consistency_20260620 \
  --output-root /experiments/a100_radial_rcs_consistency_output \
  --gpu-set 4,5,6,7 \
  --workers 8 \
  --batch-size 128
```

## OPENPAI/TOS notes for exp27

Use the full OpenPAI/TOS template in `docs/experiments/exp27-radial-rcs-consistency-gate.md`. The important runtime overrides are:

```bash
--set DATA_CONFIG.STORAGE.TYPE tos \
      DATA_CONFIG.STORAGE.INFO_TYPE tos \
      DATA_CONFIG.STORAGE.TOS.BUCKET perception-result \
      DATA_CONFIG.DATA_PATH <tos_prefix_for_highway_sweeps_raw_root> \
      DATA_CONFIG.PKL_PATH <tos_prefix_for_highway_sweeps_pkls> \
      DATA_CONFIG.INFO_PATH.train "['highway_train.pkl']" \
      DATA_CONFIG.INFO_PATH.test "['highway_test.pkl']"
```

Do not commit TOS credentials. Do not change the YAML to use storage-specific paths; the same exp27 YAML should remain valid for local A100 and TOS runs through command-line overrides.

## exp29 offline threshold/NMS scan

This is diagnosis only. It evaluates the existing exp23 checkpoint with threshold/NMS overrides and must not be treated as a formal exp29 model.

```bash
cd /workspace/worktrees/a100_radial_rcs_consistency_rebase_20260622/tools
CUDA_VISIBLE_DEVICES=4 python experiments/exp29_offline_threshold_scan.py \
  --repo /workspace/worktrees/a100_radial_rcs_consistency_rebase_20260622 \
  --root /experiments/a100_radial_rcs_consistency_20260620
```
