# exp26_radial_gate_wider_hidden

Radial gate wider-hidden variant. It uses conservative GATE_SCALE=0.35 and GATE_HIDDEN_DIM=32 to recover representation capacity without exp23 gate amplitude.

## Protocol

- Dataset: `/data/highway_sweeps/highway_train.pkl` and `/data/highway_sweeps/highway_test.pkl`.
- Sensor: `RADAR_FRONT` only. `RADAR_FRONT_TOP`, dual-front, and multi-radar fusion are not used.
- Sweeps: `MAX_SWEEPS=5`.
- ROI/evaluator: frozen 0-200m BEV/FOV and fixed HR-4D evaluator.
- Training: from scratch only; no base checkpoint fine-tuning.
- Radar-aware boundary: this experiment does not change final evaluator, does not widen the main heatmap, and does not classify all GT-box-outside wing points as background.

## A100 launch command

Do not launch formal 40-epoch training until manager approves after smoke validation.

```bash
cd /workspace/worktrees/a100_radial_rcs_consistency_20260620/tools
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train.py \
  --launcher pytorch \
  --cfg_file cfgs/radar_models/exp26_radial_gate_wider_hidden.yaml \
  --extra_tag exp26_radial_gate_wider_hidden_4gpu_40ep_20260620 \
  --batch_size 128 \
  --workers 8 \
  --logger_iter_interval 50 \
  --structured_log_iter_interval 10 \
  --ckpt_save_interval 1 \
  --fix_random_seed \
  --wo_gpu_stat
```

## Smoke command

```bash
cd /workspace/worktrees/a100_radial_rcs_consistency_20260620/tools
CUDA_VISIBLE_DEVICES=4 python tools/experiments/smoke_one_batch_radial_wing.py --cfg_file cfgs/radar_models/exp26_radial_gate_wider_hidden.yaml --batch_size 1 --workers 1
```

## Expected report requirements

- Must include baseline comparison against historical A100 single-frame baseline, exp20, exp23, and current experiment.
- Must include overall, 0-50m, 50-100m, 100-150m, 150-200m, Car, LargeVehicle/LargeV, Cyclist, Pedestrian, and pred/frame.
