# exp27_radial_rcs_consistency_gate

Radial + RCS/power temporal consistency gate. It adds current/history RCS-power delta and history variance cues, without GT point matching, evaluator changes, widened BEV, or widened main heatmap.

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
cd /workspace/worktrees/a100_radial_rcs_consistency_rebase_20260622/tools
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train.py \
  --launcher pytorch \
  --cfg_file cfgs/radar_models/exp27_radial_rcs_consistency_gate.yaml \
  --extra_tag exp27_radial_rcs_consistency_gate_4gpu_40ep_hr4d_rebase_20260622 \
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
cd /workspace/worktrees/a100_radial_rcs_consistency_rebase_20260622/tools
CUDA_VISIBLE_DEVICES=4 python experiments/smoke_one_batch_radial_wing.py --cfg_file cfgs/radar_models/exp27_radial_rcs_consistency_gate.yaml --batch_size 1 --workers 1
```

## OPENPAI/TOS Template

Use this only after the 5-sweep highway pkl files and raw `obs02` tree are available on TOS. Keep the pkl names `highway_train.pkl` and `highway_test.pkl`; do not fall back to the old single-frame pkl names. Credentials must come from the runtime environment or placeholders, never from YAML or committed files.

```yaml
commands:
  - set -euo pipefail
  - export CODE_ARCHIVE=4DRadar-HR-4D-exp27-rebase-20260622.tar.gz
  - export CODE_DIR="${CODE_ARCHIVE%.tar.gz}"
  - export EXP_NAME=exp27_radial_rcs_consistency_gate_hr4d_rebase
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
      --cfg_file cfgs/radar_models/exp27_radial_rcs_consistency_gate.yaml \
      --extra_tag "${EXP_NAME}" \
      --batch_size 256 \
      --workers 8 \
      --logger_iter_interval 50 \
      --structured_log_iter_interval 10 \
      --ckpt_save_interval 1 \
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

For an 8-GPU A100 run, `--batch_size 256` matches the later exp28-style effective batch. If reproducing the original exp27 exactly, use 4 GPUs and `--batch_size 128`, and record the batch/GPU change in the report.

## Expected report requirements

- Must include baseline comparison against historical A100 single-frame baseline, exp20, exp23, and current experiment.
- Must include overall, 0-50m, 50-100m, 100-150m, 150-200m, Car, LargeVehicle/LargeV, Cyclist, Pedestrian, and pred/frame.
- After this rebase, report the new HR-4D base commit and the rebased branch/commit used for evaluation because the evaluator strategy changed upstream.
