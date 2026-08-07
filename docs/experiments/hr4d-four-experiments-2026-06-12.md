# HR-4D Four Radar Experiments

## Record

- Date: 2026-06-12
- Owner: YJ-Y-9527
- Implementation: bladeDD, radar-dev, OpenAI Codex
- Target branch: `HR-4D`
- Upstream base reviewed: `bd0368e` (`config: add centerpoint_radar.yaml`)
- Integration commit before this document: `586fdad`
- Dataset split: `splits/hr4d_1000_v1/infos_train_800.pkl` and
  `splits/hr4d_1000_v1/infos_test_200.pkl`
- Training container: `radar-dev-4dradar-train`
- Training schedule: 80 epochs, batch size 2 per GPU, 2 GPUs per experiment

This document is the authoritative runbook for the four experiments. The
individual `docs/experiments/exp*.md` files retain the original design notes,
but their old smoke-run examples and old Git base are not the current launch
protocol.

## Compatibility Contract

The changes are additive and preserve existing HR-4D behavior:

- Existing `MeanVFE` and `PillarVFE` registrations are unchanged.
- `RadarPillarVFE` is selected only when a YAML sets
  `MODEL.VFE.NAME: RadarPillarVFE`.
- Ego-motion Doppler compensation runs only when a YAML requests the
  `ego_comp_doppler` point feature. Existing radar YAML files do not request it.
- `tools/train.py` accepts both `--local_rank` and the PyTorch 2.x spelling
  `--local-rank`; both map to the same argument.
- A clean source clone can import `pcdet` before `setup.py` generates
  `pcdet/version.py`. Installed builds still use the generated version file.
- Existing SECOND, PointPillar and CenterPoint radar YAML files load unchanged.

## Experiment Matrix

| ID | YAML | VFE | Point/voxel change | GPUs |
| --- | --- | --- | --- | --- |
| exp01 | `exp01_second_mean_xyz.yaml` | MeanVFE | XYZ only control | 9,10 |
| exp02 | `exp02_second_mean_physics.yaml` | MeanVFE | XYZ, RCS, power, Doppler, Vx, Vy, timestamp | 3,4 |
| exp03 | `exp03_second_radar_pillar_rcs.yaml` | RadarPillarVFE | exp02 features plus RCS max/std and point-count fraction | 5,6 |
| exp04 | `exp04_second_radar_pillar_ego_doppler.yaml` | RadarPillarVFE | Replaces raw Doppler with ego-compensated Doppler; aggregates RCS and Doppler max/std | 7,8 |

GPU 1 and GPU 2 are intentionally left unused. GPU 0 is used by another job.

## Code Changes

### RadarPillarVFE

Files:

- `pcdet/models/backbones_3d/vfe/radar_pillar_vfe.py`
- `pcdet/models/backbones_3d/vfe/__init__.py`

For each voxel, the VFE emits the mean of all input features. YAML-selected
feature indices additionally receive maximum and standard-deviation statistics.
An optional normalized valid-point count is appended. Padding is masked.

### Ego-Compensated Doppler

File: `pcdet/datasets/radar/radar_dataset.py`

For current frames and sweeps, vehicle speed is converted to m/s, rotated into
the current body frame, projected onto each return's line of sight, and combined
with raw Doppler. The feature is opt-in through `ego_comp_doppler` in the YAML.
Yaw-rate/lever-arm compensation is not included.

### Runtime Compatibility

Files:

- `pcdet/__init__.py`
- `tools/train.py`

These changes support clean-clone imports and PyTorch 2.x distributed launch
without changing established configuration behavior.

## Launch Commands

Run all commands inside the single container:

```bash
docker exec -it radar-dev-4dradar-train bash
cd /experiments/hr4d-integration-20260612/tools
```

Launch each experiment with two GPUs:

```bash
CUDA_VISIBLE_DEVICES=9,10 bash scripts/dist_train.sh 2 \
  --cfg_file cfgs/radar_models/exp01_second_mean_xyz.yaml \
  --extra_tag hr4d_78c1b61_2gpu

CUDA_VISIBLE_DEVICES=3,4 bash scripts/dist_train.sh 2 \
  --cfg_file cfgs/radar_models/exp02_second_mean_physics.yaml \
  --extra_tag hr4d_78c1b61_2gpu

CUDA_VISIBLE_DEVICES=5,6 bash scripts/dist_train.sh 2 \
  --cfg_file cfgs/radar_models/exp03_second_radar_pillar_rcs.yaml \
  --extra_tag hr4d_78c1b61_2gpu

CUDA_VISIBLE_DEVICES=7,8 bash scripts/dist_train.sh 2 \
  --cfg_file cfgs/radar_models/exp04_second_radar_pillar_ego_doppler.yaml \
  --extra_tag hr4d_78c1b61_2gpu
```

The current run uses one supervising shell in the same container. Logs are:

```text
/experiments/hr4d-integration-20260612/run_logs/exp01.log
/experiments/hr4d-integration-20260612/run_logs/exp02.log
/experiments/hr4d-integration-20260612/run_logs/exp03.log
/experiments/hr4d-integration-20260612/run_logs/exp04.log
```

## Monitoring

Attach to the four-window log session:

```bash
docker exec -it radar-dev-4dradar-train tmux attach -t hr4d-training
```

Use `Ctrl-b` followed by `0`, `1`, `2`, or `3` to switch experiments. Use
`Ctrl-b d` to detach without stopping training.

## Verification

Run from the repository root:

```bash
python -m unittest -q \
  tests.test_package_version \
  tests.test_radar_pillar_vfe \
  tests.test_radar_ego_doppler
```

Verified on 2026-06-12:

- Six unit tests passed.
- `second_radar.yaml`, `pointpillar_radar.yaml`, and
  `centerpoint_radar.yaml` loaded successfully.
- All four experiment YAML files loaded successfully.
- All configurations resolved the same 800/200 HR-4D split.
- The experiment commits merged with upstream `bd0368e` without conflicts.
- `git diff --check` passed after whitespace cleanup.
- Four distributed jobs entered epoch 1 successfully on GPUs 3-10.

## Reproducibility Notes

- Preserve the YAML, Git commit, split manifest, checkpoint, log and evaluation
  output for every reported metric.
- Do not compare the old 991-frame smoke results with the 800/200 protocol.
- Use the HR-4D evaluator and the same class mapping, ROI and distance buckets
  across experiments.
- The current `extra_tag` contains the initial integration base for continuity;
  the exact final Git commit must be read from the saved training log/config.
- Do not place credentials, tokens, passwords or cookies in this repository.

## Commit Inventory

- `4559824`: XYZ MeanVFE YAML baseline
- `e35a29c`: radar-physics MeanVFE YAML baseline
- `cd7c14c`: RCS-aware RadarPillarVFE and tests
- `fc84cf9`: ego-compensated Doppler and tests
- `56ec24c`: clean-clone package version fallback
- `3ffb094`: PyTorch `--local-rank` compatibility
- `62c73c1`: test whitespace cleanup
- `586fdad`: merge upstream HR-4D commit `bd0368e`
