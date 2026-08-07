# A100 radial-wing multiframe training guide

This guide is for branch `codex/a100-radial-wing-multiframe-20260619`.

## Required Data

Use the sweep dataset only:

- A100 host: `/mnt/nvme-ai-data/4DRadar/data/highway_sweeps`
- Docker: `/data/highway_sweeps`
- train pkl: `/data/highway_sweeps/highway_train.pkl`
- test pkl: `/data/highway_sweeps/highway_test.pkl`
- raw tree: `/data/highway_sweeps/obs02`

Do not use the old single-frame `/data/train_data.pkl` or
`/data/test_data.pkl` for exp20-24.

## Configs

- `cfgs/radar_models/exp20_front5_naive_stack.yaml`
- `cfgs/radar_models/exp21_front5_range_time_gate_vfe.yaml`
- `cfgs/radar_models/exp22_front5_wing_aux_gate.yaml`
- `cfgs/radar_models/exp23_front5_radial_consistency_gate.yaml`
- `cfgs/radar_models/exp24_front5_combined_candidate.yaml`

All configs preserve:

- `POINT_CLOUD_RANGE=[0,-20,-8,200,20,8]`
- `POST_CENTER_LIMIT_RANGE=[0,-20,-8,200,20,8]`
- `MIN_GT_POINTS=3`
- `EVAL_METRIC=hr4d`
- `NUM_EPOCHS=40`
- `RADAR_SENSOR=RADAR_FRONT`
- `MAX_SWEEPS=5`

## A100 Setup and Validation

Run inside `hr4d-a100-hr4d-dev`:

```bash
cd /workspace/worktrees/a100_radial_wing_multiframe_20260619
export TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda
python setup.py develop

cd tools
PYTHONPATH=/workspace/worktrees/a100_radial_wing_multiframe_20260619:/workspace/worktrees/a100_radial_wing_multiframe_20260619/tools \
python experiments/validate_radial_wing_configs.py \
  cfgs/radar_models/exp20_front5_naive_stack.yaml \
  cfgs/radar_models/exp21_front5_range_time_gate_vfe.yaml \
  cfgs/radar_models/exp22_front5_wing_aux_gate.yaml \
  cfgs/radar_models/exp23_front5_radial_consistency_gate.yaml \
  cfgs/radar_models/exp24_front5_combined_candidate.yaml
```
One-batch smoke, using one low-occupancy GPU:

```bash
cd /workspace/worktrees/a100_radial_wing_multiframe_20260619/tools
CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH=/workspace/worktrees/a100_radial_wing_multiframe_20260619:/workspace/worktrees/a100_radial_wing_multiframe_20260619/tools \
python experiments/smoke_one_batch_radial_wing.py \
  --cfg_file cfgs/radar_models/exp20_front5_naive_stack.yaml \
  --batch_size 1 --workers 1
```

Repeat the smoke command for exp21-24 before formal training.

## Durable Queue Launch

The preferred long-run entrypoint is the supervisor. It serializes exp20-24 on
GPUs 4-7 and writes state, heartbeat, child logs, reports and Feishu send logs.

```bash
mkdir -p /experiments/a100_radial_wing_multiframe_20260619/logs
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
echo $! > /experiments/a100_radial_wing_multiframe_20260619/supervisor.pid
```

Read-only status:

```bash
cd /workspace/worktrees/a100_radial_wing_multiframe_20260619/tools
python experiments/a100_radial_wing_supervisor.py \
  --root /experiments/a100_radial_wing_multiframe_20260619 \
  --status
```

## Feishu Notification

The supervisor reads `FEISHU_BOT_WEBHOOK` and `FEISHU_BOT_SECRET` only from the
process environment or root-only env files such as `/root/.codex/feishu_bot.env`.
Do not write credential values into configs, docs, logs or memory.

## OPENPAI/TOS Template

This suite is currently validated on the A100 local sweep mount. For OpenPAI,
use the template below only after the sweep pkl and raw `obs02` tree have known
TOS prefixes. Replace placeholders; do not use single-frame pkl names.

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
