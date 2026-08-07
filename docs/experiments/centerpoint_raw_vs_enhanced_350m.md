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
| raw | n/a | pending GPU run | — | — | blocked: NVIDIA device unavailable in current session |
| sr-0 | neighborhood filling, threshold 0.5, `add_offset=True` | pending GPU run | — | — | blocked: NVIDIA device unavailable in current session |

No mAP value is fabricated.  Once the GPU device is visible, run the two
commands above with identical options and append the actual evaluation output;
the SR branch is accepted only when `enhanced_mAP - raw_mAP >= 0.050`.
