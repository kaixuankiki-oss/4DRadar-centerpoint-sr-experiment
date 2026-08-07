# Experiment 22: RADAR_FRONT 5-sweep wing-aware BEV gate

## Objective

Use lateral-wing point statistics as auxiliary BEV evidence without widening the
BEV range or the main CenterHead heatmap.

## Hypothesis

Lateral wing returns are often object-associated rather than pure clutter. A
small BEV feature gate using per-pillar radar statistics can keep this evidence
available while preserving the frozen detection target geometry.

## Model Change

- Config: `tools/cfgs/radar_models/exp22_front5_wing_aux_gate.yaml`
- MAP_TO_BEV: `WingAwarePointPillarScatter`
- Extra per-pillar stats: range, RCS, absolute Doppler, timestamp and point
  count.
- The stats produce a residual gate on BEV features; grid size and CenterHead
  targets remain unchanged.

## Frozen Protocol

- Data: `/data/highway_sweeps/highway_train.pkl`,
  `/data/highway_sweeps/highway_test.pkl`, raw tree `/data/highway_sweeps/obs02`.
- `RADAR_FRONT`, `MAX_SWEEPS=5`, `MIN_GT_POINTS=3`.
- `POINT_CLOUD_RANGE=[0,-20,-8,200,20,8]`
- `POST_CENTER_LIMIT_RANGE=[0,-20,-8,200,20,8]`
- `EVAL_METRIC=hr4d`, `NUM_EPOCHS=40`, train from scratch.
- Do not use old single-frame `/data/train_data.pkl` or `/data/test_data.pkl`.

## A100 Validation

```bash
cd /workspace/worktrees/a100_radial_wing_multiframe_20260619
export TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda
python setup.py develop

cd tools
PYTHONPATH=/workspace/worktrees/a100_radial_wing_multiframe_20260619:/workspace/worktrees/a100_radial_wing_multiframe_20260619/tools \
python experiments/validate_radial_wing_configs.py \
  cfgs/radar_models/exp22_front5_wing_aux_gate.yaml

CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH=/workspace/worktrees/a100_radial_wing_multiframe_20260619:/workspace/worktrees/a100_radial_wing_multiframe_20260619/tools \
python experiments/smoke_one_batch_radial_wing.py \
  --cfg_file cfgs/radar_models/exp22_front5_wing_aux_gate.yaml \
  --batch_size 1 --workers 1
```

## A100 Training Command

Normally launch through `experiments/a100_radial_wing_supervisor.py` from
`exp20-front5-naive-stack.md` so exp20-24 run serially on GPUs 4-7.

Manual single-experiment fallback:

```bash
cd /workspace/worktrees/a100_radial_wing_multiframe_20260619/tools
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train.py \
  --launcher pytorch \
  --cfg_file cfgs/radar_models/exp22_front5_wing_aux_gate.yaml \
  --extra_tag exp22_front5_wing_aux_gate_4gpu_40ep_20260619 \
  --batch_size 128 \
  --workers 8 \
  --logger_iter_interval 50 \
  --structured_log_iter_interval 10 \
  --ckpt_save_interval 1 \
  --fix_random_seed \
  --wo_gpu_stat
```

## OPENPAI/TOS Template

Use the same template as exp20, replacing only `EXP_NAME` and `--cfg_file`.
The sweep pkl names must remain `highway_train.pkl` and `highway_test.pkl`.
