# Frame200 CenterPoint A/B experiment (0–350 m)

## Scope

The raw and reconstructed branches use the same CenterPoint configuration,
random seed, train/validation frame IDs, optimizer, voxelization, augmentor,
batch size and epoch count.  The only changed input is the `RADAR_FRONT`
`radar_path` in the generated info PKL:

- raw: `frame200_ori/.../radar_front_bottom/*.pcd`
- reconstructed: `output/radar_front_bottom_sr/.../*_SR.pcd`

Both branches use single-frame points and the five features
`x,y,z,RCS,AbsV` (the PCD's canonical field spelling for the requested
`x,y,z,rcs,absv`).  Ground-truth boxes come directly from
`frame200_ori/infos_test_200.pkl`.

## Data split

`tools/experiments/prepare_frame200_centerpoint.py` creates 160 train and 40
validation samples (every fifth source frame is validation).  The manifest is
`centerpoint_data/frame200/manifest.pkl`.  No PCD files are copied.

## Range and model settings

The A/B config is
`tools/cfgs/radar_models/centerpoint_frame200_350m.yaml`:

- `POINT_CLOUD_RANGE = [0, -20, -8, 350, 20, 8]`
- CenterHead post-center range uses the same limits
- HR4D `EVAL_RANGE = [0, 350]`
- `MAX_SWEEPS = 1`
- voxel size `[0.25, 0.2, 16.0]` (XY grid `[1400, 200]`, divisible by 16)

For the reconstructed branch, use the same config and override only the two
info paths:

```bash
python tools/train.py --cfg_file tools/cfgs/radar_models/centerpoint_frame200_350m.yaml \
  --extra_tag frame200_raw_350m --fix_random_seed

python tools/train.py --cfg_file tools/cfgs/radar_models/centerpoint_frame200_350m.yaml \
  --extra_tag frame200_sr_350m --fix_random_seed \
  --set DATA_CONFIG.INFO_PATH.train /home/kaixuan-ding/Workspace/point_cloud_ob/centerpoint_data/frame200/sr_train.pkl \
        DATA_CONFIG.INFO_PATH.test /home/kaixuan-ding/Workspace/point_cloud_ob/centerpoint_data/frame200/sr_val.pkl
```

Evaluate the selected checkpoints with `tools/test.py` and record the HR4D
`mAP` values below.  Because this experiment has only 200 labelled frames,
the required comparison is now an absolute difference of at least `+0.050`
(five percentage points, not a relative percentage).

## Results

| run | fill strategy | checkpoint | mAP | absolute delta vs raw | status |
|---|---|---|---:|---:|---|
| raw | n/a | `checkpoint_epoch_40.pth` | 0.194115 | 0.000000 | completed; baseline |
| sr-0 | neighborhood filling, threshold 0.5, `add_offset=True` | `checkpoint_epoch_35.pth` | 0.042826 | -0.151289 | failed; replaced in place by sr-1 |
| sr-1 | exact raw preservation, threshold 0.8, SR RCS ≥ 2 dB | `checkpoint_epoch_40.pth` | 0.036061 | -0.158054 | failed; ROI pillars diluted by generated points |
| sr-2 | raw preservation + dynamic empty-voxel SR gate | `checkpoint_epoch_38.pth` | 0.038440 | -0.155675 | failed; LargeVehicle/Cyclist AP regressed |
| sr-3 | sr-2 + high-RCS low-speed empty-voxel points | `checkpoint_epoch_39.pth` | 0.034671 | -0.159444 | failed; static clutter increased |
| sr-4 | sr-3 gate restricted to 0–50 m additions | `checkpoint_epoch_40.pth` | 0.061758 | -0.132357 | failed; best SR result so far |
| sr-5 | sr-4 + dynamic raw-support longitudinal expansion | `checkpoint_epoch_37.pth` | 0.109287 | -0.084828 | failed; best enhanced run so far, Cyclist AP recovered to 0.217 |
| sr-6 | dynamic + dense-slow raw support; learned candidates disabled | `checkpoint_epoch_39.pth` | 0.206328 | +0.012213 | improved over raw, but below required +0.050 |
| sr-7 | sr-6 with PCA-oriented dense-slow expansion | `checkpoint_epoch_35.pth` | 0.209153 | +0.015038 | improved; both Cyclists recalled, ordinary vehicle AP regressed |
| sr-8 | sr-7 PCA orientation gated by eigenvalue ratio ≥ 10 | `checkpoint_epoch_38.pth` | 0.058631 | -0.135484 | failed; small direction substitutions destabilized training |
| sr-9 | sr-6 longitudinal support + selective PCA lateral additions | `checkpoint_epoch_40.pth` | 0.054984 | -0.139131 | failed; sr-7 remains best |
| sr-10 | dynamic support + PCA-selected lateral dense support only | `checkpoint_epoch_40.pth` | 0.184553 | -0.009562 | failed; both Cyclists recalled but vehicle AP regressed |
| sr-11 | sr-6 + internal interpolation for lateral dense clusters | `checkpoint_epoch_37.pth` | 0.055280 | -0.138835 | failed; sr-7 remains best |
| sr-12 | sr-6 fixed-axis dense gate relaxed to 6 points/RCS 2 | `checkpoint_epoch_38.pth` | 0.060416 | -0.133699 | failed; relaxed gate did not recover convergence |
| sr-13 | sr-7 + second lateral voxel for PCA ratio ≥ 10 | `checkpoint_epoch_31.pth` | 0.049283 | -0.144832 | failed; extra lateral pillar occupancy destroyed late convergence |
| sr-14 | sr-7 geometry with synthetic-support RCS × 0.5 | `checkpoint_epoch_37.pth` | 0.162134 | -0.031981 | failed; lower synthetic RCS weakened vehicle evidence |
| sr-15 | sr-7 geometry with synthetic-support RCS × 1.5 | `checkpoint_epoch_40.pth` | 0.038535 | -0.155580 | failed; amplified support removed Cyclist detections |
| sr-16 | sr-7 geometry, dynamic RCS × 1.0 and dense RCS × 0.5 | `checkpoint_epoch_40.pth` | 0.041811 | -0.152304 | failed; provenance-aware scaling did not recover convergence |
| sr-17 | sr-7 geometry with source-cell-relative synthetic coordinates | `checkpoint_epoch_40.pth` | 0.046188 | -0.147927 | failed; intra-voxel offset copy did not preserve convergence |
| sr-18 | sr-7 plus anisotropy-aware dynamic support | `checkpoint_epoch_36.pth` | 0.181160 | -0.012955 | failed; dynamic PCA still lost vehicle/Cyclist balance |
| sr-19 | sr-6 plus lateral PCA only for dense seeds at 40–45m | `checkpoint_epoch_37.pth` | 0.046777 | -0.147338 | failed; range-only orientation gate still destabilized convergence |
| sr-20 | sr-6 plus PCA lateral support at 42–43.5m, RCS 5–12 | `checkpoint_epoch_40.pth` | 0.046709 | -0.147406 | failed; narrow range/RCS gate still destabilized convergence |
| sr-21 | sr-6 with dynamic raw-support gate `|AbsV|>=3.0` (dense support unchanged) | `checkpoint_epoch_34.pth` | 0.180762 | -0.013353 | failed; stricter dynamic gate removed useful support |
| sr-22 | sr-6 geometry/gates with synthetic-support AbsV × 0.5 | `checkpoint_epoch_36.pth` | 0.036039 | -0.158076 | failed; global AbsV scaling removed Cyclist detections |
| sr-23 | sr-6 with dynamic source min 2 raw points + adjacent dynamic source requirement | `checkpoint_epoch_40.pth` | 0.046941 | -0.147174 | failed; dynamic support became too sparse |
| sr-24 | sr-6 with adjacent dynamic-source requirement, min 1 raw point | `checkpoint_epoch_36.pth` | 0.037460 | -0.156655 | failed; neighbor gate still removed convergence |
| sr-25 | sr-6 dynamic support with positive-only longitudinal expansion | `checkpoint_epoch_40.pth` | 0.202895 | +0.008780 | failed; direction gate retained Cyclist but no Car gain |
| sr-26 | sr-6 support + learned SR points at threshold 0.99, RCS ≥ 2, range < 50m | `checkpoint_epoch_39.pth` | 0.054355 | -0.139760 | failed; sparse learned geometry destabilized training |
| sr-27 | sr-6 support + high-RCS static raw expansion (RCS ≥ 25) | `checkpoint_epoch_37.pth` | 0.186979 | -0.007136 | failed; static additions reduced vehicle AP |
| sr-28 | sr-6 support + ultra-high-RCS static raw expansion (RCS ≥ 30) | `checkpoint_epoch_40.pth` | 0.0433658 | -0.1507493 | failed; sparse static additions still changed convergence |
| sr-29 | sr-7 PCA dense support with dense synthetic RCS × 0.8 | running | — | — | active; raw/dynamic features unchanged |

No mAP value is fabricated.  GPU access is available through the host execution
environment (the default Codex sandbox intentionally hides `/dev/nvidia*`).
The RTX 4070 Ti SUPER passed PyTorch CUDA and both raw/SR CenterPoint one-batch
forward/backward tests on 2026-08-07.  The SR branch is accepted only when
`enhanced_mAP - raw_mAP >= 0.050`.

## Host-side job control

Inference and training are launched in a detached process group so they
survive an SSH disconnect. From the repository root, the physical host can
control the active stage with:

```bash
tools/experiments/frame200_job_control.sh status
tools/experiments/frame200_job_control.sh pause
tools/experiments/frame200_job_control.sh resume
tools/experiments/frame200_job_control.sh log 100
tools/experiments/frame200_job_control.sh stop
```

`pause` sends `SIGSTOP` to the whole process group and creates a persistent
pause marker; `resume` removes the marker and sends `SIGCONT`. `stop` requests
a graceful `SIGTERM`. Runtime PID/log/control files live under ignored
`.experiment_control/` and are not committed.
