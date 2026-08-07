# Experiment 20: RADAR_FRONT 5-sweep naive stack

## Objective

Establish the scratch-trained 5-sweep `RADAR_FRONT` baseline before adding any
range/time/wing-aware model logic.

## Hypothesis

Using the current frame plus four history sweeps should increase sparse far-range
evidence without changing BEV size or CenterHead target geometry. This tests
whether the sweep dataset alone improves 100-200m detection.

## Code and Config

- Config: `tools/cfgs/radar_models/exp20_front5_naive_stack.yaml`
- Model: base CenterPoint/PillarVFE/PointPillarScatter.
- Data: `/data/highway_sweeps/highway_train.pkl`,
  `/data/highway_sweeps/highway_test.pkl`, raw tree `/data/highway_sweeps/obs02`.
- Sensor/sweeps: `RADAR_FRONT`, `MAX_SWEEPS=5`.

## Frozen Protocol

- `POINT_CLOUD_RANGE=[0,-20,-8,200,20,8]`
- `POST_CENTER_LIMIT_RANGE=[0,-20,-8,200,20,8]`
- `MIN_GT_POINTS=3`
- `EVAL_METRIC=hr4d`
- `NUM_EPOCHS=40`
- Train from scratch; do not load baseline checkpoints.
- Do not use old single-frame `/data/train_data.pkl` or `/data/test_data.pkl`.

## A100 Validation

Run from the A100 container `hr4d-a100-hr4d-dev`:

```bash
cd /workspace/worktrees/a100_radial_wing_multiframe_20260619
export TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda
python setup.py develop

cd tools
PYTHONPATH=/workspace/worktrees/a100_radial_wing_multiframe_20260619:/workspace/worktrees/a100_radial_wing_multiframe_20260619/tools \
python experiments/validate_radial_wing_configs.py \
  cfgs/radar_models/exp20_front5_naive_stack.yaml

CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH=/workspace/worktrees/a100_radial_wing_multiframe_20260619:/workspace/worktrees/a100_radial_wing_multiframe_20260619/tools \
python experiments/smoke_one_batch_radial_wing.py \
  --cfg_file cfgs/radar_models/exp20_front5_naive_stack.yaml \
  --batch_size 1 --workers 1
```
## A100 Training Command

Preferred queue launch uses the durable supervisor so train/eval/report/Feishu
are tracked as child jobs:

```bash
cd /workspace/worktrees/a100_radial_wing_multiframe_20260619/tools
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONPATH=/workspace/worktrees/a100_radial_wing_multiframe_20260619:/workspace/worktrees/a100_radial_wing_multiframe_20260619/tools \
nohup python experiments/a100_radial_wing_supervisor.py \
  --repo /workspace/worktrees/a100_radial_wing_multiframe_20260619 \
  --root /experiments/a100_radial_wing_multiframe_20260619 \
  --output-root /experiments/a100_radial_wing_multiframe_output \
  --gpu-set 4,5,6,7 \
  --batch-size 128 \
  --workers 8 \
  > /experiments/a100_radial_wing_multiframe_20260619/logs/supervisor.out 2>&1 &
```

Manual single-experiment fallback:

```bash
cd /workspace/worktrees/a100_radial_wing_multiframe_20260619/tools
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train.py \
  --launcher pytorch \
  --cfg_file cfgs/radar_models/exp20_front5_naive_stack.yaml \
  --extra_tag exp20_front5_naive_stack_4gpu_40ep_20260619 \
  --batch_size 128 \
  --workers 8 \
  --logger_iter_interval 50 \
  --structured_log_iter_interval 10 \
  --ckpt_save_interval 1 \
  --fix_random_seed \
  --wo_gpu_stat
```

## OPENPAI/TOS Template

Only use this after the sweep pkl and raw `obs02` tree are available in TOS.
Replace the placeholder prefixes; do not fall back to single-frame pkl names.

```yaml
commands:
  - set -euo pipefail
  - export CODE_ARCHIVE=4DRadar-HR-4D.tar.gz
  - export CODE_DIR="${CODE_ARCHIVE%.tar.gz}"
  - export EXP_NAME=exp20_front5_naive_stack
  - bash /mnt/data-vepfs/token/huoshan-tos.sh
  - mkdir -p /mnt/nas && cd /mnt/nas
  - rm -rf "${CODE_DIR}"
  - tosutil cp -r tos://e2e-training/code/${PAI_USER_NAME}/openpcdet/${CODE_ARCHIVE} .
  - tar -zxf "${CODE_ARCHIVE}"
  - source ~/miniconda/etc/profile.d/conda.sh
  - conda activate 4d
  - export TORCH_CUDA_ARCH_LIST="8.0"
  - export CUDA_HOME=/usr/local/cuda
  - export NCCL_IB_DISABLE=1
  - export NCCL_SOCKET_IFNAME=eth0
  - export GLOO_SOCKET_IFNAME=eth0
  - export TOS_AK=<your_tos_access_key>
  - export TOS_SK=<your_tos_secret_key>
  - export TOS_ENDPOINT=tos-cn-shanghai.ivolces.com
  - export TOS_REGION=cn-shanghai
  - cd "/mnt/nas/${CODE_DIR}"
  - pip install setuptools==58.2.0
  - python setup.py develop
  - cd tools
  - |
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py \
      --launcher pytorch \
      --cfg_file cfgs/radar_models/exp20_front5_naive_stack.yaml \
      --extra_tag "${EXP_NAME}" \
      --batch_size 128 \
      --workers 8 \
      --fix_random_seed \
      --wo_gpu_stat \
      --set DATA_CONFIG.STORAGE.TYPE tos \
            DATA_CONFIG.STORAGE.INFO_TYPE tos \
            DATA_CONFIG.STORAGE.TOS.BUCKET perception-result \
            DATA_CONFIG.DATA_PATH <tos_prefix_for_highway_sweeps_raw_root> \
            DATA_CONFIG.PKL_PATH <tos_prefix_for_highway_sweeps_pkls> \
            DATA_CONFIG.INFO_PATH.train "['highway_train.pkl']" \
            DATA_CONFIG.INFO_PATH.test "['highway_test.pkl']"
```
