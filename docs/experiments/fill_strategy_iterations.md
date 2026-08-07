# Reconstructed point-cloud fill strategy log

Every strategy writes to the same `output/radar_front_bottom_sr` root and the
same `_SR.pcd` name.  Re-running a strategy therefore replaces the previous
PCD in place; it does not create a versioned copy.  The corresponding Center-
Point info files are regenerated after each replacement.

## sr-0 — current baseline

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

## Iteration template

When a measured SR mAP is below raw mAP + 0.020:

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
