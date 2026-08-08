# Radar SR inference snapshot

This directory is the version-controlled snapshot of the point-cloud SR code
used by the Frame200 CenterPoint experiment. The working copy is executed from
`/home/kaixuan-ding/Workspace/point_cloud_ob`; after each matching-strategy
change, the files here are refreshed before committing.

The model checkpoint is intentionally not committed. Pass it explicitly:

```bash
python tools/experiments/radar_sr/reconstructed_inference.py \
  --model_path /path/to/model_epoch_60.pth \
  --input_dir /path/to/frame200_ori --recursive \
  --output_dir /path/to/output/radar_front_bottom_sr
```

For the exact arguments used by each experiment, see
`docs/experiments/fill_strategy_iterations.md`.
