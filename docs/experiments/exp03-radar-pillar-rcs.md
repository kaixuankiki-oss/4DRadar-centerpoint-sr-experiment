# Experiment 03: RadarPillarVFE with RCS statistics

## Objective

Preserve strong-scatterer information that MeanVFE suppresses by averaging all
points in a voxel.

## Hypothesis

Radar targets are often represented by a small number of dominant scattering
centres. RCS maximum and standard deviation should complement the ordinary
feature mean, while point-count fraction provides a local density cue.

## Code Change

`RadarPillarVFE` emits:

1. Mean of every input feature over valid points.
2. Maximum RCS.
3. RCS standard deviation.
4. Valid point count divided by the configured voxel capacity.

Padding is explicitly masked and cannot affect mean, max or standard deviation.
The output remains one feature vector per voxel, so the SECOND backbone and
detection head are unchanged.

## Controlled Variables

- Input features: identical to Experiment 02
- Detector, voxelization, temporal sweeps, classes and optimizer: unchanged
- VFE change only: MeanVFE to RadarPillarVFE
- Random seed: fixed by `--fix_random_seed`

## Configuration

`tools/cfgs/radar_models/exp03_second_radar_pillar_rcs.yaml`

## Verification

```bash
python -m unittest tests.test_radar_pillar_vfe
cd tools
python train.py \
  --cfg_file cfgs/radar_models/exp03_second_radar_pillar_rcs.yaml \
  --batch_size 1 --epochs 1 --workers 0 \
  --extra_tag smoke_f772y8 --fix_random_seed --wo_gpu_stat \
  --set DATA_CONFIG.DATA_PATH /data/1000_original_data \
        OUTPUT_ROOT /experiments
```

## Results

Pending unit test and smoke run.
