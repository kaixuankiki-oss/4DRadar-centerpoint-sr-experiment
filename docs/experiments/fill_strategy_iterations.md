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
- Full overwrite inference, PKL regeneration, and fixed-config training
  completed. Actual inventory was 18,893 generated points with 23.47% labelled
  box-hit rate. Best result: epoch 40, mAP `0.1845533887`, absolute delta
  `-0.0095617080`; failed. Both Cyclists were recalled, but Car/LargeVehicle AP
  fell to `0.081`/`0.028`, confirming ordinary longitudinal dense support was
  responsible for much of sr-6's vehicle gain.

## sr-11 — sr-6 plus internal dense-cluster bridges

- Motivation: retain sr-6 exactly for vehicle performance, then strengthen the
  slow lateral Tricycle without expanding its outer footprint.
- Code change: added `--bridge_dense_raw` and `dense_bridge_*`. For ratio-10
  lateral dense-slow clusters, nearby qualifying seed cells (within 1.5 m) are
  connected by discrete voxel interpolation; only raw-empty cells between
  observed endpoints are filled.
- Offline audit: 1,152 bridge candidates over 200 frames, including three
  points inside the missed validation Tricycle and no labelled Car or
  LargeVehicle boxes. This is intentionally a geometry-continuity operation,
  not outward dilation.
- sr-6 longitudinal dynamic/dense expansion remains unchanged and learned-SR
  candidates remain disabled.
- Full overwrite inference, PKL regeneration, and fixed-config training
  completed. Best result: epoch 37, mAP `0.0552797885`, absolute delta
  `-0.1388353082`; failed. Internal bridging did not preserve sr-6's late
  convergence, so sr-7 remains the best enhanced run.

## sr-12 — relaxed fixed-axis dense support

- Motivation: run a direct support-strength ablation without another geometry
  algorithm. sr-6's fixed longitudinal dense support produced the best vehicle
  AP; increasing its coverage may also make the second slow Tricycle robust.
- Parameters changed from sr-6 only: dense cell minimum 8 -> 6 points and
  representative RCS minimum 5 -> 2. Adaptive orientation and bridging are
  disabled; learned additions remain disabled.
- Offline estimate: about 22,009 low-speed longitudinal candidates over 200
  frames and roughly 11 additions inside the second validation Tricycle.
- Full overwrite inference, PKL regeneration, and fixed-config training
  completed. Best result: epoch 38, mAP `0.0604159499`, absolute delta
  `-0.1336991467`; failed. Relaxing the dense gate from 8/RCS5 to 6/RCS2 did
  not recover the sr-6/sr-7 convergence.

## sr-13 — conservative strong-lateral extension

- Motivation: retain sr-7's complete input distribution, which is currently
  the best enhanced result, while adding only a small amount of support to
  thin lateral targets. The previous sr-8/sr-9 tests changed or added lateral
  support for every selected PCA cell and destabilized training.
- Code change: added `dense_expand_lateral_steps` and
  `dense_expand_lateral_min_ratio`. sr-13 keeps sr-7's thresholds and PCA
  selection, but adds the second lateral voxel only when the local PCA ratio
  is at least 10. Raw points, dynamic expansion, and all other hyperparameters
  remain unchanged.
- Data handling: inference overwrites the shared 970-file `_SR.pcd` set;
  PKLs are rebuilt from those files before training. No labels are read by
  the filling code.
- Full overwrite inference, PKL regeneration, and fixed-config training
  completed. The best checkpoint was epoch 31 with mAP `0.049283`, absolute
  delta `-0.144832` versus raw; failed. Even the ratio-10-only second lateral
  voxel altered enough pillar occupancy to destroy the late convergence, so
  adding extra points to sr-7 is rejected.

## sr-14 — RCS-attenuated sr-7 support

- Motivation: sr-13 showed that changing occupancy is especially harmful on
  this 200-frame split. sr-14 keeps sr-7's point locations and count exactly,
  and changes only the copied RCS value on synthetic raw-support points.
- Code change: added `raw_expand_rcs_scale` and `raw_expand_absv_scale`.
  The raw points are untouched; sr-14 uses RCS scale `0.5` and AbsV scale
  `1.0` for dynamic and dense additions. Learned SR candidates remain gated
  off, and `add_offset=True` remains explicit.
- Full overwrite inference, PKL rebuild, and fixed-config training completed.
  Best result: epoch 37, mAP `0.1621342172`, absolute delta
  `-0.0319808795`; failed. Cyclist AP remained `0.3193`, but the lower
  synthetic RCS reduced Car/LargeVehicle evidence. Since scale `1.0` in sr-7
  measured `0.209153`, the next ablation tests the other direction without
  changing geometry or occupancy.

## sr-15 — RCS-amplified sr-7 support (running)

- Motivation: the controlled feature-only sweep shows mAP increasing from
  scale `0.5` (`0.162134`) to scale `1.0` (`0.209153`). sr-15 tests whether
  stronger synthetic support can cross the required `0.244115` threshold.
- Strategy: exact sr-7 point locations, point count, gates, PCA orientation,
  split, config and seed; only synthetic raw-support RCS changes to scale
  `1.5`. Raw `x,y,z,RCS,AbsV` values remain untouched and learned candidates
  remain disabled.
- Full overwrite inference, PKL rebuild, and fixed-config training completed.
  Best result: epoch 40, mAP `0.0385348013`, absolute delta
  `-0.1555802954`; failed. Amplifying support removed both Cyclist detections,
  confirming that feature strength must be separated by support type.

## sr-16 — dynamic/full-strength plus attenuated dense support

- Motivation: sr-14 attenuated both dynamic and dense points and reached
  `0.162134`; sr-15 amplified both and collapsed to `0.038535`. sr-16 keeps
  dynamic high-confidence support at RCS scale `1.0` and attenuates only the
  low-speed dense support to `0.5`.
- Code change: `best_seed_by_target` now retains the support provenance
  (`dynamic` or `dense`), enabling separate feature scaling without changing
  coordinates, occupancy or point count. Raw observations remain exact.
- Full overwrite inference, PKL rebuild, and fixed-config training completed.
  Best result: epoch 40, mAP `0.0418110522`, absolute delta
  `-0.1523040444`; failed. Provenance-aware scaling did not recover the
  sr-7 convergence, despite leaving dynamic additions at full strength.

## sr-17 — source-cell-relative coordinates

- Motivation: all RCS feature ablations changed convergence. sr-17 restores
  sr-7's exact gates and feature values, but replaces the synthetic point's
  grid-center coordinate with the source raw point's intra-voxel offset copied
  into the target cell. This preserves realistic within-pillar geometry while
  keeping occupancy and count unchanged.
- Code change: added `raw_expand_coordinate_mode={center,copy_offset}`;
  raw points are untouched and the fill remains label-independent.
- Full overwrite inference, PKL rebuild, and fixed-config training completed.
  Best result: epoch 40, mAP `0.0461883126`, absolute delta
  `-0.1479267840`; failed. Copying intra-voxel offsets did not preserve the
  late convergence, so sr-7's grid-center placement remains preferable.

## sr-18 — anisotropy-aware dynamic support

- Motivation: sr-7 only orients dense slow cells. High-confidence dynamic raw
  cells are still expanded along a fixed longitudinal axis, which can widen
  cross-traffic vehicles in the wrong direction.
- Planned code change: add a label-independent local-PCA option for dynamic
  seeds, gated by a strong eigenvalue ratio (10) and falling back exactly to
  sr-7's longitudinal offsets when the cluster is not strongly anisotropic.
  Dynamic and dense feature values, point count, output root, split and
  CenterPoint settings remain unchanged.
- Full overwrite inference, PKL rebuild, and fixed-config training completed.
  Best result: epoch 36, mAP `0.1811603636`, absolute delta
  `-0.0129547330`; failed. The dynamic PCA gate improved over several feature
  ablations but still lost sr-7's vehicle/Cyclist balance.

## sr-19 — range-isolated lateral correction

- Motivation: the missed stationary validation Tricycle's two dense cells are
  around 42–43m with local PCA ratios above 10. sr-19 keeps sr-6's strong
  longitudinal vehicle support everywhere, and permits lateral PCA orientation
  only for dense seeds in the label-independent 40–45m band.
- This narrows sr-7's orientation change from thousands of cells to the small
  feature/range-defined subset around the observed cross-traffic geometry;
  all raw points and feature values remain exact.
- Full overwrite inference, PKL rebuild, and fixed-config training completed.
  Best result: epoch 37, mAP `0.0467770365`, absolute delta
  `-0.1473380602`; failed. Even the 40–45m range gate changed enough dense
  cells to lose convergence, so the next test narrows by both range and RCS.

## sr-20 — narrow RCS/range lateral correction

- Motivation: the missed stationary Tricycle's two PCA cells have ranges
  42.19/42.81m and representative RCS 10.21/7.18. sr-20 retains sr-6's
  longitudinal support except for strong-PCA dense cells in range 42–43.5m
  and representative RCS 5–12, minimizing unrelated orientation changes.
- The gate uses only raw point count, RCS, AbsV, range and local occupancy; no
  annotations are read by inference.
- Full overwrite inference, PKL rebuild, and fixed-config training completed.
  Best result: epoch 40, mAP `0.0467088749`, absolute delta
  `-0.1474062218`; failed. Even the two-dimensional feature gate did not
  retain sr-6 convergence, so orientation changes are abandoned.

## sr-21 — higher-confidence dynamic support

- Motivation: retain sr-6's fixed longitudinal geometry and dense-slow support,
  but remove lower-confidence dynamic expansions. The offline audit raises
  dynamic candidate box-hit rate from 26.3% at `|AbsV|>=1.5` to 28.9% at
  `|AbsV|>=3`, while the first validation Cyclist still receives 33 supports;
  the stationary Cyclist remains supplied by dense-slow support.
- Only `raw_expand_min_abs_v` changes from sr-6 (`1.5 -> 3.0`). All feature
  values, coordinates, point count rules, split, seed and CenterPoint settings
  are otherwise identical.
- Full overwrite inference, PKL rebuild and 40-epoch training completed
  through the host controller. The shared output contains 970 freshly
  rewritten `_SR.pcd` files. Best checkpoint: epoch 34, mAP
  `0.1807617954`, absolute delta `-0.0133533013` versus raw; failed. The
  stricter dynamic gate removed useful support and did not preserve sr-6's
  convergence.

## sr-22 — dense-support feature-strength ablation

- Motivation: sr-21 confirms that reducing dynamic candidate count alone is
  not enough. The next controlled test keeps sr-6 geometry and candidate
  gates, but attenuates only the copied AbsV on synthetic raw-support points
  to reduce pillar-feature distortion while leaving RCS, coordinates and
  occupancy unchanged.
- Only `raw_expand_absv_scale` changes from sr-6 (`1.0 -> 0.5`); raw points,
  dynamic/dense geometry, learned-SR gates, split, seed and CenterPoint
  settings remain unchanged. Inference remains label-independent and uses
  `add_offset=True`.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 36,
  mAP `0.0360394645`, absolute delta `-0.1580756322` versus raw; failed.
  Scaling AbsV on both dynamic and dense synthetic points changed the input
  distribution enough to remove both Cyclist detections.

## sr-23 — locally consistent dynamic raw support (planned)

- Motivation: offline, label-independent auditing shows that isolated high-
  Doppler raw cells are much noisier than dynamic cells with local support.
  Require each dynamic source voxel to contain at least two raw returns and
  to have another qualifying dynamic source within one XY voxel. This keeps
  the existing longitudinal expansion and all synthetic features unchanged,
  while removing isolated clutter.
- New controls: `dynamic_expand_min_points=2` and
  `dynamic_expand_require_neighbor=True`; all sr-6 thresholds, dense-slow
  support, coordinates, feature values, split, seed and CenterPoint settings
  remain unchanged. Inference remains label-independent with `add_offset=True`.
- Offline audit: 5,604 dynamic candidates at a 46.7% labelled-box hit rate,
  versus 16,829 candidates at 26.3% for sr-6; the first validation Cyclist
  retains 27 generated supports and the stationary Cyclist remains supplied
  by dense-slow support.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 40,
  mAP `0.0469410521`, absolute delta `-0.1471740446` versus raw; failed.
  The precision gain was outweighed by removing too much dynamic support;
  both Cyclist detections disappeared despite the offline support audit.

## sr-24 — adjacent dynamic support without source-density gate (planned)

- Motivation: sr-23's combined two-point and neighbor gate was too sparse for
  the detector to converge. Keep the higher-precision local-neighbor rule,
  but restore isolated single-return source cells so the dynamic candidate
  inventory is larger while still rejecting isolated clutter.
- Only `dynamic_expand_min_points` changes from sr-23 (`2 -> 1`); the
  adjacent-dynamic requirement, all dense-slow support, feature values,
  geometry, split, seed and CenterPoint settings remain unchanged.
- Offline audit estimates 7,128 dynamic candidates at 39.5% box-hit rate and
  29 supports in the first validation Cyclist; stationary Cyclist support is
  unchanged.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 36,
  mAP `0.0374597487`, absolute delta `-0.1566553480` versus raw; failed.
  Neighbor consistency without the density gate still removed the late
  convergence seen in sr-6.

## sr-25 — positive-direction dynamic expansion (planned)

- Motivation: the offline audit shows that the positive longitudinal neighbor
  has higher labelled-box precision (`30.1%`) than the negative neighbor
  (`25.0%`). Keep sr-6's full dynamic source inventory and dense-slow support,
  but add only the positive-x target cell for each dynamic seed to reduce the
  lower-precision half of the fill.
- New control: `dynamic_expand_direction=positive`; no consistency gate is
  enabled. All features, range/RCS/AbsV thresholds, dense support, split,
  seed and CenterPoint settings remain unchanged.
- Offline audit: 9,199 dynamic candidates at 30.1% box-hit rate versus 16,829
  at 26.3% for the two-sided sr-6 rule; the first validation Cyclist retains
  22 supports, while dense-slow support remains unchanged.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 40,
  mAP `0.2028953257`, absolute delta `+0.0087802290` versus raw; failed.
  Positive-only support preserved both Cyclist detections and LargeVehicle AP,
  but did not improve the Car AP obtained by sr-6.

## sr-26 — sparse high-confidence learned SR additions (planned)

- Motivation: the learned occupancy model can identify additional empty cells,
  but the earlier learned-SR runs admitted thousands of low-quality points.
  A single-frame diagnostic at `threshold=0.99`, matched `RCS>=2` and range
  below 50m retained only 45 learned points (after empty-voxel and geometry
  gates), making a sparse test possible.
- Keep sr-6 deterministic dynamic/dense support and add only these learned
  points. Set `threshold=0.99`, `sr_min_rcs=2`, omit the learned AbsV gate,
  and keep `sr_max_range=50`; raw points, features, coordinates, split, seed
  and CenterPoint settings remain unchanged. `add_offset=True` remains
  explicit and inference does not read labels.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 39,
  mAP `0.0543547741`, absolute delta `-0.1397603226` versus raw; failed.
  Sparse learned geometry still destabilized the small-data training and
  reduced all three useful class APs.

## sr-27 — high-RCS static raw support (planned)

- Motivation: offline auditing found a small, high-precision source of static
  vehicle evidence that sr-6 does not use. Expanding only low-speed source
  voxels with representative RCS ≥ 25 dB adds about 688 candidates at a
  22.2% box-hit rate, including 149 Car box hits; RCS ≥ 30 yields 237
  candidates at 44.7% hit rate. This targets Car AP without changing the
  dynamic/dense support that preserves Cyclist recall.
- New controls: `expand_strong_static_raw=True`,
  `strong_static_min_rcs=25`, `strong_static_min_points=1`,
  `strong_static_max_abs_v=0.5`, `strong_static_max_range=50`. All sr-6
  dynamic/dense rules, feature values, geometry, split, seed and CenterPoint
  settings remain unchanged; inference is label-independent with
  `add_offset=True`.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 37,
  mAP `0.1869790351`, absolute delta `-0.0071360616` versus raw; failed.
  Cyclist recall was preserved, but the RCS-25 additions reduced Car and
  LargeVehicle AP relative to sr-6.

## sr-28 — ultra-high-RCS static support

- Motivation: sr-27's 688 estimated static additions were still enough to
  disturb convergence. Raise the static threshold to 30 dB, where the offline
  candidate inventory falls to about 237 points with a 44.7% labelled-box hit
  rate (106 Car hits and no LargeVehicle/Cyclist hits).
- Only `strong_static_min_rcs` changes from sr-27 (`25 -> 30`). All sr-6
  dynamic/dense rules, point features, geometry, split, seed and CenterPoint
  settings remain unchanged.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 40,
  mAP `0.0433658046`, absolute delta `-0.1507492921` versus raw; failed.
  Even the sparse RCS≥30 additions changed the small-data training balance,
  so this branch is not retained as the preferred strategy.

## sr-29 — PCA dense support with provenance-aware RCS attenuation

- Motivation: sr-7 is the strongest measured branch, while prior global
  attenuation changed both dynamic and dense evidence together. This test
  keeps sr-7 geometry and gates, but attenuates only synthetic dense-support
  RCS to `0.8`; raw points and dynamic synthetic points remain unchanged.
  The hypothesis is that dense slow returns should provide occupancy without
  overpowering measured dynamic evidence.
- Only `dense_expand_rcs_scale=0.8` is new. Learned SR points remain disabled
  (`sr_min_rcs=1e9`), raw points are preserved exactly, `add_offset=True`,
  split/seed/config/range are unchanged, and inference does not read labels.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 35,
  mAP `0.139397`, absolute delta `-0.0547180967` versus raw; failed. The
  attenuation changed the learned balance even though only dense points were
  scaled, so this branch is discarded.

## sr-30 — PCA dense support with voxel-median feature matching

- Motivation: sr-29 shows that a global dense RCS scale is not a reliable
  feature match. For each dense slow source voxel, compute the median raw RCS
  and AbsV over that voxel and assign those statistics to synthetic target
  cells. This suppresses an isolated peak without applying a fixed global
  scale. Dynamic support keeps exact source-point RCS/AbsV and all sr-7
  geometry/gates remain unchanged.
- New controls: `dense_expand_feature_mode=voxel_median` (with the default
  source mode for dynamic/other support). Learned SR remains disabled,
  `add_offset=True`, labels are not read by inference, and the split/seed/
  CenterPoint config/range are unchanged.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 39,
  mAP `0.199499`, absolute delta `+0.0053839033` versus raw; failed. Offline
  statistics show that the median reduced dense RCS substantially, so the
  branch retained only a small gain over raw.

## sr-31 — PCA dense support with voxel-quantile feature matching

- Motivation: retain the robust AbsV median from sr-30 while restoring more
  of the dense return amplitude. The dense source voxel now supplies its RCS
  75th percentile and AbsV median; dynamic and strong-static supports still
  copy the selected raw source point. This is an adaptive statistic, not a
  fixed global scale.
- New controls: `dense_expand_feature_mode=voxel_quantile` and
  `dense_expand_feature_quantile=0.75`. All geometry, gates, learned-SR
  disablement, `add_offset=True`, split, seed, CenterPoint settings and
  label-independent inference constraints remain unchanged.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 38,
  mAP `0.190676`, absolute delta `-0.0034390967` versus raw; failed. The
  upper-quantile dense RCS did not recover the vehicle balance.

## sr-32 — high-confidence dynamic gate with PCA dense support

- Motivation: offline audit gives the dynamic `|AbsV|>=2.5, RCS>=10` rule a
  28.16% labelled-box hit rate, higher than sr-6's 26.32%, while preserving
  33 generated supports in the first validation Cyclist. Use this as an
  unlabeled precision gate; dense slow support remains the sr-7 PCA rule with
  exact source features so the stationary Cyclist is not removed.
- Only `raw_expand_min_abs_v` changes from sr-7 (`1.5 -> 2.5`). Learned SR is
  disabled, raw points remain exact, `add_offset=True`, and all split/seed/
  CenterPoint/range settings remain fixed.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 36,
  mAP `0.173334`, absolute delta `-0.0207810967` versus raw; failed. Car and
  LargeVehicle AP both regressed, so the stricter dynamic gate is discarded.

## sr-33 — provenance-specific PCA-lateral RCS support

- Motivation: sr-7's main advantage is detecting both validation Cyclists,
  but its Cyclist AP is still only 0.484. Separate PCA-lateral dense support
  from ordinary dense support and apply a mild RCS factor only to that
  provenance. Dynamic points and ordinary dense points retain exact source
  features, avoiding the global scaling failure seen in sr-15.
- New control: `dense_expand_adaptive_rcs_scale=1.25`; the implementation
  tags PCA-lateral proposals as `dense_adaptive` and otherwise preserves the
  established sr-7 behavior. No annotations enter inference, learned SR is
  disabled, `add_offset=True`, and all A/B settings remain fixed.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 31,
  mAP `0.093709`, absolute delta `-0.1004060967` versus raw; failed. Scaling
  only PCA-lateral supports still destabilized the small-data optimization.

## sr-34 — sr-7 geometry with dynamic voxel-median feature matching

- Motivation: sr-7 remains the strongest geometric branch. Dynamic source
  voxels have a much smaller max-to-median RCS gap than dense voxels, so replace
  only dynamic synthetic RCS/AbsV with voxel medians while preserving the
  dense-PCA source features that recover both Cyclists.
- New control: `raw_expand_feature_mode=voxel_median`; dense mode explicitly
  remains `source`. All gates, geometry, `add_offset=True`, label-independent
  inference, split, seed, CenterPoint settings and 0–350m range are fixed.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 32,
  mAP `0.181259`, absolute delta `-0.0128560967` versus raw; failed. Dynamic
  voxel medians reduced LargeVehicle AP, so exact source features are restored.

## sr-35 — positive dynamic support plus PCA dense support

- Motivation: combine sr-25's lower-clutter positive-only dynamic geometry
  with sr-7's PCA-oriented dense support. Positive dynamic targets have higher
  offline box-hit precision than negative targets, while PCA dense support is
  the only tested rule that raised Cyclist AP above the raw value.
- Only `dynamic_expand_direction=positive` changes from sr-7. Source features,
  dense PCA rules, learned-SR disablement, `add_offset=True`, split, seed,
  CenterPoint settings and range are unchanged.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 32,
  mAP `0.140337`, absolute delta `-0.0537780967` versus raw; failed. The
  combination reduced LargeVehicle and Cyclist AP.

## sr-36 — half-scale local support grid

- Motivation: offline audit at a `0.125 x 0.10m` grid raises dynamic candidate
  box-hit rate from 26.32% to 29.67% and increases in-box coverage. The
  half-step targets remain close to measured returns and may reinforce pillar
  features without widening objects as aggressively as the standard grid.
- Change `raw_expand_voxel_size` from `0.25 x 0.20m` to `0.125 x 0.10m` and
  scale `dense_expand_min_points` from 8 to 4 for the smaller cell area. Keep
  sr-7 PCA orientation, exact source features, all gates, `add_offset=True`,
  split/seed/config and label-independent inference unchanged.
- Full overwrite inference, PKL rebuild and 40-epoch training completed. The
  shared 970-file output was replaced in place. Best checkpoint: epoch 40,
  mAP `0.029490`, absolute delta `-0.1646250967` versus raw; failed. The
  smaller source grid produced too few dense supports and destabilized the
  detector.

## sr-37 — sparse high-confidence learned SR plus sr-7 support

- Motivation: deterministic raw support remains stable but cannot recover all
  missing occupancy. Re-enable only very high-confidence learned points at
  threshold `0.995`, matched RCS≥2 and range<50m, while retaining sr-7 dynamic
  and PCA dense support. The higher threshold is intended to avoid the broad
  learned-SR distribution that failed in sr-26.
- Only learned occupancy filtering changes: `threshold=0.995`,
  `sr_min_rcs=2`, no AbsV learned-point gate, `sr_max_range=50`; raw points,
  deterministic features/geometry, `add_offset=True`, split, seed and
  CenterPoint settings remain fixed. Inference remains label-independent.
- Full overwrite inference, PKL rebuild and fixed-config 40-epoch training
  completed. The shared 970 outputs were overwritten in place. The best
  checkpoint was epoch 38 with mAP `0.046289`, absolute delta
  `-0.1478261867` versus raw; failed. Even the `.995` learned occupancy gate
  removed the Cyclist detections, so learned SR is disabled again.

## sr-38 — sr-7 geometry with dense-voxel mean feature matching

- Motivation: sr-30's dense-voxel median matching under-attenuated the
  representative RCS and reached `0.199499`, while sr-31's p75/median mix did
  not recover the vehicle classes. This round keeps the strongest sr-7
  geometry and dynamic source features, but assigns synthetic dense-slow
  points the arithmetic mean RCS and AbsV of their raw source voxel. The mean
  is a less aggressive estimator than the median for multi-return vehicle
  cells while removing the single maximum-return bias.
- Only `dense_expand_feature_mode=voxel_mean` changes from sr-7. Learned SR is
  disabled (`threshold=0.8`, impossible learned RCS gate), raw points remain
  exact, `add_offset=True`, single-frame x/y/z/RCS/AbsV, 0–350m range, split,
  seed and CenterPoint hyperparameters remain fixed. Inference is
  label-independent.
- Full overwrite inference, PKL rebuild and fixed-config 40-epoch training
  completed. The shared 970 `_SR.pcd` paths were overwritten in place. The
  final best checkpoint was epoch 40 with mAP `0.040986`, absolute delta
  `-0.1531290347` versus raw; failed. The dense mean weakened the learned
  pillar evidence and removed Cyclist detections.

## sr-39 — sr-7 support with intermediate PCA anisotropy gate

- Motivation: sr-7 switches dense-slow support to lateral expansion whenever
  the local PCA major axis is lateral (`min_axis_ratio=1`), recovering both
  validation Cyclists but lowering vehicle AP. The ratio-10 variant (sr-8)
  was too restrictive and also failed to converge. This round tests the
  intermediate, label-independent ratio `2`, retaining more longitudinal
  vehicle support while still allowing clearly anisotropic lateral clusters.
- Only `dense_expand_min_axis_ratio` changes from sr-7 (`1 -> 2`). Source
  features, point coordinates, learned-SR disablement, `add_offset=True`,
  single-frame features, 0–350m range, split/seed and CenterPoint settings
  remain unchanged. Inference does not read labels.
- Full overwrite inference, PKL rebuild and fixed-config 40-epoch training
  completed. The 970 shared outputs were overwritten in place. Best
  checkpoint: epoch 35, mAP `0.190519`, absolute delta `-0.0035960290`
  versus raw; failed. Car/LargeVehicle/Cyclist AP were
  `0.100400/0.037824/0.433333`, so ratio 2 lost sr-7's Cyclist gain without
  recovering enough vehicle AP.

## sr-40 — local PCA gate search at ratio 1.25

- Motivation: the strongest endpoint is ratio 1 (sr-7, `0.209153`), while
  ratio 2 (sr-39, `0.190519`) restores the raw Cyclist AP and loses overall
  performance. Search near the successful endpoint with ratio `1.25` to
  retain weakly anisotropic lateral support while reverting only nearly
  isotropic dense clusters to longitudinal vehicle support.
- Only `dense_expand_min_axis_ratio` changes from sr-7 (`1 -> 1.25`). Exact
  source feature matching, all gates/coordinates, learned-SR disablement,
  `add_offset=True`, single-frame inputs, 0–350m range, split/seed and
  CenterPoint hyperparameters remain fixed. Inference is label-independent.
- Full overwrite inference verified `970/970` fresh outputs, followed by PKL
  rebuild and fixed-config 40-epoch training. Best checkpoint: epoch 30, mAP
  `0.190688`, absolute delta `-0.0034274513` versus raw; failed. Class AP was
  Car `0.092654`, LargeVehicle `0.046076`, Cyclist `0.433333`. The near-ratio
  gate still removed sr-7's Cyclist gain, so PCA threshold search is stopped.

## sr-41 — attenuated high-confidence learned-SR feature matching

- Motivation: sr-26/sr-37 show that even high-confidence learned occupancy
  destroys Cyclist convergence when learned points receive full interpolated
  RCS. Preserve sr-37's sparse geometry, but attenuate only learned-point RCS
  to `0.25`; deterministic sr-7 source features and all raw points remain
  exact. This tests whether learned occupancy can contribute geometry without
  dominating PointPillar max-pooled features.
- Code change: add `learned_sr_rcs_scale` and `learned_sr_absv_scale` controls
  applied after learned-point filtering and empty-voxel deduplication, before
  merging deterministic support. Defaults are 1.0 for backward compatibility.
- sr-41 uses threshold `.995`, learned matched RCS≥2, range<50m,
  `learned_sr_rcs_scale=.25`, AbsV scale 1.0, plus exact sr-7 dynamic/PCA
  support. Inference remains label-independent; `add_offset=True`, input
  features, 0–350m range, split/seed/config are unchanged.
- Full overwrite inference verified `970/970` fresh outputs, followed by PKL
  rebuild and fixed-config 40-epoch training. Best checkpoint: epoch 36, mAP
  `0.039375`, absolute delta `-0.1547400007`; failed. Car/LargeVehicle/Cyclist
  AP were `0.090771/0.027354/0.000000`. Learned geometry still removed all
  Cyclist detections despite learned-only RCS attenuation, so learned SR is
  disabled again.

## sr-42 — dense source-voxel median z matching (running)

- Motivation: all prior deterministic feature tests copied the z of the
  highest-RCS seed, even for dense voxels containing at least eight raw
  returns. That single height can be an outlier and is duplicated into an
  adjacent pillar. Use the median raw z in the source voxel for dense and
  PCA-dense synthetic support only; keep sr-7 x/y occupancy and source
  RCS/AbsV unchanged.
- Code change: add `dense_expand_z_mode={source,voxel_median,voxel_mean}`;
  defaults preserve existing behavior. Values are computed only from raw
  source-voxel points, without annotations or learned outputs.
- sr-42 sets `dense_expand_z_mode=voxel_median` on exact sr-7 geometry/gates.
  Learned SR stays disabled; original points remain exact; `add_offset=True`,
  single-frame x/y/z/RCS/AbsV, 0–350m, split/seed/config remain fixed.
- The shared 970 output paths will be overwritten before PKL rebuild/training.
