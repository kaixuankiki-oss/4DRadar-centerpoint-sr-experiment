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
- Full overwrite inference and PKL regeneration completed. The raw-checkpoint
  diagnostic was `0.1781634497`, but the required independent training best
  was epoch 38, mAP `0.0384404497`, absolute delta `-0.1556746470`; failed.
  Class AP was Car `0.087`, LargeVehicle `0.029`, Cyclist `0.000`.

## sr-3 — dynamic OR strong-static empty-voxel filling

- Motivation: sr-2 retained Car AP near raw but over-filtered low-speed/static
  LargeVehicle returns. It also remained unstable on the two evaluated
  Cyclist instances.
- Code change: added `--sr_static_min_rcs`. A generated point passes the
  motion gate when `|AbsV| >= 1.5` **or** `RCS >= 15 dB`; the existing
  `RCS >= 2 dB`, raw-empty voxel, and one-point-per-voxel constraints remain.
- This is label-independent at inference time. It aims to recover strong
  stationary vehicle surfaces without reintroducing weak static clutter.
- Full overwrite inference, PKL regeneration, and fixed-config training
  completed. Best result: epoch 39, mAP `0.0346710438`, absolute delta
  `-0.1594440529`; failed.

## sr-4 — near-range-only gated filling

- Motivation: sr-3 generated-point precision by forward-distance band was
  23.3% at 0–50 m, 6.0% at 50–100 m, 3.7% at 100–150 m, and below 2.4%
  beyond 150 m. Far additions were mostly clutter.
- Code change: added `--sr_min_range` / `--sr_max_range` and set
  `sr_max_range=50`. All raw points remain present throughout 0–350 m; only
  generated additions are range-gated. Other sr-3 gates remain unchanged.
- Full overwrite inference, PKL regeneration, and fixed-config training
  completed. Best result: epoch 40, mAP `0.0617577823`, absolute delta
  `-0.1323573144`; failed, but materially better than sr-1 through sr-3.

## sr-5 — near dynamic raw-support expansion

- Motivation: the learned SR model does not produce matched additions around
  every dynamic small target. Raw points within 50 m satisfying
  `|AbsV| >= 1.5` and `RCS >= 10 dB` have a measured box-hit rate of 47.1%.
- Code change: added `--expand_dynamic_raw` and `raw_expand_*` controls. Each
  qualifying original return proposes the immediately adjacent longitudinal
  XY cells; occupied raw cells are skipped and each target cell keeps the
  highest-RCS source. Attributes are copied from that source. The expansion
  uses only point features, not annotations.
- Train-only/offline geometry audit: the two-direction expansion produces
  16,829 points over all 200 frames with a 26.3% box-hit rate after empty-cell
  deduplication, and adds support near both evaluated Tricycle boxes.
- sr-4 learned additions and raw-support expansions are merged to at most one
  generated point per `0.25 x 0.20 m` cell.
- Full overwrite inference, PKL regeneration, and fixed-config training
  completed. Best result: epoch 37, mAP `0.1092866455`, absolute delta
  `-0.0848284512`; failed, but became the best enhanced run. Class AP was Car
  `0.088`, LargeVehicle `0.024`, and Cyclist `0.217`.
- The 200 labelled frames contained 33,739 generated points with a 19.4% box
  hit rate. The first in-range validation Tricycle received 39 generated
  points, while the second received none because its raw returns had
  `|AbsV|=0.01–0.11`.

## sr-6 — dynamic plus dense-slow raw support, no learned additions

- Motivation: sr-5 recovered part of Cyclist AP, but its learned-SR additions
  reduced candidate box precision relative to dynamic expansion alone. The
  missed in-range Tricycle already has 37 raw returns concentrated in nine XY
  cells, but is effectively stationary and cannot pass a dynamic gate.
- Added `tools/experiments/analyze_frame200_expansion.py`. It applies candidate
  rules without annotations, then uses annotations only to audit precision and
  per-class coverage. No annotation is read by inference.
- Code change: added `--expand_dense_raw` and `dense_expand_*`. A slow cell is
  expanded longitudinally when it has at least 8 raw returns, its maximum-RCS
  representative has `RCS >= 5` and `|AbsV| < 0.5`, and range is below 50 m.
  The existing dynamic rule remains in parallel.
- Learned-SR candidates are disabled with an unreachable RCS gate. Model
  inference still runs with `add_offset=True`; all raw points remain exact.
- Offline audit estimate over 200 frames: dynamic expansion contributes 16,829
  candidates at 26.3% box-hit rate; dense-slow expansion contributes at most
  about 10,510 candidates and gives the previously missed validation
  Tricycle four generated support points. Collisions are deduplicated.
- Reproducible pipeline: `tools/experiments/run_frame200_sr6.sh` overwrites the
  same 970 PCD paths, verifies every output was rewritten, rebuilds PKLs, and
  runs the identical fixed-seed 40-epoch CenterPoint training.
- Full overwrite inference, PKL regeneration, and fixed-config training
  completed. All 970 files passed the fresh-overwrite check. Best result:
  epoch 39, mAP `0.2063282826`, absolute delta `+0.0122131859`. This is the
  first enhanced run to exceed raw, but it remains `0.0377868141` below the
  required target. Class AP was Car `0.123`, LargeVehicle `0.062`, and Cyclist
  `0.433`.

## sr-7 — PCA-oriented dense-slow support

- Motivation: inspecting sr-6 predictions showed that the missed Tricycle has
  nearby detections, but they are classified as Car. The nearest prediction is
  0.64 m from the GT center with dimensions about `4.1 x 1.85 m`, while the GT
  is a roughly 83-degree rotated `2.83 x 1.30 m` box. Fixed longitudinal
  expansion widens this laterally oriented target in the wrong direction.
- Code change: added `--dense_expand_adaptive_axis` and
  `--dense_expand_axis_radius`. For each qualifying dense-slow seed, PCA is
  computed from qualifying cell centers within 3 m. Expansion uses the
  cardinal XY axis closest to the local principal axis. Dynamic strong-return
  expansion remains longitudinal.
- Label-independent audit over 200 frames: 11,495 adaptive dense candidates,
  2.05% labelled-box hit rate, and four additions in the missed validation
  Tricycle. The count is comparable to sr-6; the material change is support
  orientation rather than point volume.
- Full overwrite inference, PKL regeneration, and fixed-config training
  completed. Best result: epoch 35, mAP `0.2091531481`, absolute delta
  `+0.0150380514`; improved over sr-6 but still below target. Cyclist AP rose
  to `0.484` with recall@4m `1.0`, proving that the second target became
  detectable. Car/LargeVehicle AP fell to `0.109`/`0.034`, showing that
  unconditional PCA orientation changed too many ordinary vehicle clusters.

## sr-8 — anisotropy-gated PCA orientation

- Motivation: preserve sr-6's Car/LargeVehicle gain while retaining sr-7's
  second-Cyclist recall.
- Code change: added `--dense_expand_min_axis_ratio`. A dense-slow cell changes
  from longitudinal to lateral expansion only when its local PCA major/minor
  eigenvalue ratio is at least 10; otherwise it uses the sr-6 rule.
- The missed Tricycle's two qualifying cells have ratios `14.26` and `11.20`,
  so both retain lateral support. Across 200 frames only 1,122 qualifying
  cells choose lateral expansion, versus 3,033 in sr-7.
- Label-independent audit: 10,817 dense candidates, 2.23% labelled-box hit
  rate, and four additions in the missed validation Tricycle.
- Full overwrite inference, PKL regeneration, and fixed-config training
  completed. Best result: epoch 38, mAP `0.0586310396`, absolute delta
  `-0.1354840571`; failed. Despite retaining the two target-cell orientations,
  replacing even this subset of longitudinal supports destabilized the small
  training set and removed the late convergence seen in sr-6/sr-7.

## sr-9 — sr-6 support plus selective lateral additions

- Motivation: keep the complete sr-6 input distribution and add, rather than
  substitute, lateral evidence for the missed slow Tricycle.
- Code change: added `--dense_expand_keep_longitudinal`. When the ratio-10 PCA
  gate selects a lateral cluster, its two longitudinal cells remain and two
  lateral cells are added, forming a cross. Other cells are exactly sr-6.
- This changes fewer cells than sr-7 and gives the missed validation Tricycle
  about eight generated supports instead of four.
- Full overwrite inference, PKL regeneration, and fixed-config training
  completed. Best result: epoch 40, mAP `0.0549837012`, absolute delta
  `-0.1391313954`; failed. Additive lateral support also removed the late
  convergence, so sr-7 remains the best enhanced run.

## sr-10 — dynamic plus PCA-selected dense support only

- Motivation: remove ordinary static dense expansions and retain only the two
  label-independent signals with the clearest object evidence.
- Code change: added `--dense_expand_require_adaptive_axis`. With it enabled,
  dense-slow seeds that do not pass the ratio-10 lateral PCA gate are skipped.
- Strategy: learned additions disabled; dynamic `|AbsV| >= 1.5, RCS >= 10`
  longitudinal support retained; only strongly anisotropic lateral slow
  clusters receive dense support. This is substantially sparser than sr-6 to
  sr-9 while still reaching the second validation Tricycle.
- Full overwrite inference, PKL regeneration, and fixed-config training:
  pending.
