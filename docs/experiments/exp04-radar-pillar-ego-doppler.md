# Experiment 04: RadarPillarVFE with ego-compensated Doppler

## Objective

Separate target radial motion from the radial velocity induced by vehicle
translation, then preserve both RCS and compensated-Doppler distributions in
each voxel.

## Hypothesis

Raw Doppler entangles target motion with ego motion. A stationary object in
front of a vehicle moving at 10 m/s is measured near -10 m/s. Adding the ego
velocity projected onto the radar line of sight should move static returns
toward zero and make motion features easier for the detector to learn.

## Code Change

For every current frame and historical sweep:

1. Convert `vehiclespeed` from km/h to m/s.
2. Apply the forward or reverse sign from `vehicledirection`.
3. Rotate the sweep-body velocity into the current body frame.
4. Project ego velocity onto each return's radar line of sight.
5. Add that projection to raw Doppler as `ego_comp_doppler`.

`RadarPillarVFE` appends max and standard-deviation statistics for both RCS
and ego-compensated Doppler, plus the voxel point-count fraction. Yaw-rate
compensation is deliberately excluded until its unit and sensor lever-arm
convention are verified.

## Controlled Variables

- Git parent: Experiment 03 commit `fc802db`
- Detector, voxelization, sweeps, classes and optimizer: unchanged
- Input feature count: unchanged; raw Doppler is replaced by compensated Doppler
- VFE adds compensated-Doppler max/std to Experiment 03 output
- Random seed: fixed by `--fix_random_seed`

## Configuration

`tools/cfgs/radar_models/exp04_second_radar_pillar_ego_doppler.yaml`

## Verification

```bash
python -m unittest tests.test_radar_pillar_vfe tests.test_radar_ego_doppler
cd tools
python train.py \
  --cfg_file cfgs/radar_models/exp04_second_radar_pillar_ego_doppler.yaml \
  --batch_size 1 --epochs 1 --workers 0 \
  --extra_tag smoke_f772y8 --fix_random_seed --wo_gpu_stat \
  --set DATA_CONFIG.DATA_PATH /data/1000_original_data \
        OUTPUT_ROOT /experiments
```

The current train/test PKL is shared. Results below validate execution, not
generalization.

## Results

Pending unit tests and smoke run.
