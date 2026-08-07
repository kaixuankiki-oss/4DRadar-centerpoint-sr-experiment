# Experiment 01: MeanVFE with XYZ only

## Objective

Establish the geometry-only radar baseline before introducing radar physics.
This experiment measures how much detection performance comes from point
geometry alone.

## Hypothesis

Removing RCS, power, Doppler, velocity and sweep time reduces the input signal
available to the detector. The result is the control group for later
physics-aware experiments.

## Controlled Variables

- Git base: `ae3de5a`
- Detector: SECONDNet with AnchorHeadSingle
- VFE: MeanVFE
- Point features: `x, y, z`
- Radar sensor: `RADAR_FRONT`
- Temporal input: current frame plus three sweeps
- Voxel size: `[0.2, 0.2, 0.4]`
- Point cloud range: `[-120, -80, -8, 260.8, 80, 8]`
- Classes: Vehicle, Pedestrian, Cyclist
- Random seed: fixed by `--fix_random_seed`

## Configuration

`tools/cfgs/radar_models/exp01_second_mean_xyz.yaml`

## Smoke Training

```bash
cd tools
python train.py \
  --cfg_file cfgs/radar_models/exp01_second_mean_xyz.yaml \
  --batch_size 1 --epochs 1 --workers 0 \
  --extra_tag smoke_f772y8 --fix_random_seed --wo_gpu_stat \
  --set DATA_CONFIG.DATA_PATH /data/1000_original_data \
        OUTPUT_ROOT /experiments
```

The 1000-frame PKL currently uses the same source for train and test. Smoke
metrics verify execution only and must not be treated as benchmark results.

## Results

Pending smoke run.
