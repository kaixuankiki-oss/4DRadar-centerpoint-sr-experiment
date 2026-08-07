# Experiment 02: MeanVFE with radar physics

## Objective

Measure the gain from directly encoding radar scattering and motion features
without changing the aggregation method.

## Hypothesis

RCS, power, Doppler, planar velocity and sweep time contain class, motion and
temporal information that XYZ alone cannot represent. MeanVFE may recover part
of this value while exposing the limitations of mean-only aggregation.

## Controlled Variables

- Git base: `ae3de5a`
- Detector: SECONDNet with AnchorHeadSingle
- VFE: MeanVFE
- Point features: `x, y, z, RCS, power, doppler, Vx, Vy, timestamp`
- Radar sensor: `RADAR_FRONT`
- Temporal input: current frame plus three sweeps
- Voxel size, range, classes and optimizer: identical to Experiment 01
- Random seed: fixed by `--fix_random_seed`

## Configuration

`tools/cfgs/radar_models/exp02_second_mean_physics.yaml`

## Smoke Training

```bash
cd tools
python train.py \
  --cfg_file cfgs/radar_models/exp02_second_mean_physics.yaml \
  --batch_size 1 --epochs 1 --workers 0 \
  --extra_tag smoke_f772y8 --fix_random_seed --wo_gpu_stat \
  --set DATA_CONFIG.DATA_PATH /data/1000_original_data \
        OUTPUT_ROOT /experiments
```

The current train/test source is shared, so this run validates the pipeline
rather than producing a publishable metric.

## Results

Pending smoke run.
