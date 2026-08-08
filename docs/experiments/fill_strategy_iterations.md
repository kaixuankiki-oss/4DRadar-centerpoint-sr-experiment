# Reconstructed point-cloud fill strategy log

Every strategy writes to the same `output/radar_front_bottom_sr` root and the
same `_SR.pcd` name.  Re-running a strategy therefore replaces the previous
PCD in place; it does not create a versioned copy.  The corresponding Center-
Point info files are regenerated after each replacement.

## sr-0 — baseline (failed)

- Motivation: establish the existing implementation as the A/B baseline.
- `reconstructed_inference.py`: no matching rule change.
- Matching: distance-first range/azimuth/elevation lookup, neighborhood
  filling enabled, RCS weighted by bin distance, Doppler/AbsV representative
  assignment, AAI overlay enabled.
- Inference: `threshold=0.5`, `add_offset=True`, `is_super_resolution=True`.
- Full recursive input inventory: 970 raw PCDs; output root contains 970
  corresponding `_SR.pcd` files.
- CenterPoint status: data-loader and GPU one-batch forward/backward smoke tests
  passed for both raw and SR data.  The default Codex sandbox hides GPU device
  nodes, so full CUDA jobs must use the approved host execution environment.
- Measured best result: epoch 35, mAP `0.0428261293`; raw best mAP is
  `0.1941150967`, giving an absolute delta of `-0.1512889674`.

## Iteration template

When a measured SR mAP is below raw mAP + 0.050:

1. Record the hypothesis and exact function/arguments changed here.
2. Re-run recursive inference with the same output root, which overwrites
   `_SR.pcd` files in place.
3. Re-run `prepare_frame200_centerpoint.py` to rebuild matched info PKLs.
4. Train raw and SR with the identical 350m config and fixed split.
5. Evaluate both on the same 40 validation frames and append measured mAP and
   absolute delta to `centerpoint_raw_vs_enhanced_350m.md`.

Because only 200 labelled frames are available, do not claim success unless
the measured absolute delta is at least `+0.050` (five absolute percentage
points).

## sr-1 — exact raw preservation plus high-confidence SR points

- Motivation: sr-0 reconstructed/reprojected original measurements through
  AAI/bin coordinates and reduced the RCS distribution. Preserve every raw
  observation exactly so an SR iteration cannot discard the baseline signal.
- Code change: `reconstructed_inference.py` adds
  `--preserve_original_points` and `--sr_min_rcs`. Original
  `x,y,z,RCS,AbsV` values are copied directly; generated points at an exact
  scaled original bin are deduplicated; generated points below the RCS gate
  are rejected.
- Inference arguments: `threshold=0.8`, `add_offset=True`, `use_aai=False`,
  `use_original_overlay=False`, `use_neighborhood_filling=True`,
  `preserve_original_points=True`, `sr_min_rcs=2.0`.
- All 970 outputs replaced sr-0 in the same
  `output/radar_front_bottom_sr/**/*_SR.pcd` paths. No versioned PCD copy was
  retained. The identical 160/40 info split was regenerated successfully.
- CenterPoint run tag: `frame200_sr1_350m` (40 epochs, fixed seed and same
  config as raw). Best result: epoch 40, mAP `0.0360612459`, absolute delta
  `-0.1580538508`; failed.

## sr-2 — dynamic, empty-voxel-only filling

- Motivation: within the CenterPoint ROI, sr-1 added 278,221 generated points
  (20.9% of the raw count). Although 11.4% of generated points were inside a
  labelled box, generated points shared existing pillars and changed their
  capped/averaged features. The raw checkpoint evaluated directly on sr-1 at
  only `0.0239477`, confirming a severe input-distribution shift.
- Offline analysis (used to choose a label-independent rule): requiring
  `|AbsV| >= 1.5`, filling only raw-empty `0.25 x 0.20 m` XY voxels, and keeping
  only the maximum-RCS SR point per voxel reduces the candidate inventory to
  about 18,897 points across 200 labelled frames. About 21.2% of these points
  fall inside a labelled box, versus 3.3% for raw ROI points.
- Code change: added `--sr_min_abs_v` and `--sr_empty_voxel_size`. The rule
  uses only inference-time point features/occupancy; it does not read labels.
- Arguments retained from sr-1 plus `sr_min_abs_v=1.5` and
  `sr_empty_voxel_size=(0.25, 0.20)`.
- Full overwrite inference, PKL regeneration, and fixed-config training:
  in progress.
