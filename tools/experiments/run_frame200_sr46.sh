#!/usr/bin/env bash
set -euo pipefail

# sr-46: sr-45 low dynamic RCS plus dynamic-only AbsV x1.5. PCA-dense
# features remain at full source strength.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="/home/kaixuan-ding/Workspace/point_cloud_ob"
PYTHON="/home/kaixuan-ding/miniconda3/envs/spconv/bin/python"
RAW_ROOT="${WORKSPACE}/frame200_ori"
SR_ROOT="${WORKSPACE}/output/radar_front_bottom_sr"
INFO_ROOT="${WORKSPACE}/centerpoint_data/frame200"
CONTROL_ROOT="${REPO_ROOT}/.experiment_control/frame200_sr"
MARKER="${CONTROL_ROOT}/sr46_inference_started"

mkdir -p "${CONTROL_ROOT}"
touch "${MARKER}"
trap 'rm -f "${MARKER}"' EXIT

echo "stage=sr46_inference status=starting timestamp=$(date --iso-8601=seconds)"
cd "${WORKSPACE}"
"${PYTHON}" reconstructed_inference.py \
    --input_dir "${RAW_ROOT}" --recursive \
    --output_dir "${SR_ROOT}" \
    --model_path "${WORKSPACE}/model_epoch_60.pth" \
    --model_type zynq --threshold 0.8 --add_offset true \
    --use_aai false --use_original_overlay false \
    --add_is_sr true --use_neighborhood_filling true \
    --preserve_original_points true \
    --sr_min_rcs 1000000000 --sr_min_abs_v 1.5 \
    --sr_static_min_rcs 15 --sr_max_range 50 \
    --sr_empty_voxel_size 0.25 0.20 \
    --expand_dynamic_raw true \
    --raw_expand_min_abs_v 1.5 --raw_expand_min_rcs 10 \
    --raw_expand_max_range 50 --raw_expand_voxel_size 0.25 0.20 \
    --dynamic_expand_rcs_scale 0.25 \
    --dynamic_expand_absv_scale 1.5 \
    --expand_dense_raw true \
    --dense_expand_min_points 8 --dense_expand_min_rcs 5 \
    --dense_expand_max_abs_v 0.5 --dense_expand_max_range 50 \
    --dense_expand_adaptive_axis true --dense_expand_axis_radius 3 \
    --dense_expand_rcs_scale 1.0 --dense_expand_absv_scale 1.0

expected_count="$(find "${RAW_ROOT}" -type f -path '*/radar_front_bottom/*.pcd' | wc -l)"
output_count="$(find "${SR_ROOT}" -type f -name '*_SR.pcd' | wc -l)"
stale_count="$(find "${SR_ROOT}" -type f -name '*_SR.pcd' ! -newer "${MARKER}" | wc -l)"
echo "stage=sr46_inference status=verified expected=${expected_count} outputs=${output_count} stale=${stale_count}"
if [[ "${output_count}" -ne "${expected_count}" || "${stale_count}" -ne 0 ]]; then
    echo "sr-46 inference inventory verification failed" >&2
    exit 1
fi

echo "stage=sr46_info_rebuild status=starting timestamp=$(date --iso-8601=seconds)"
cd "${REPO_ROOT}"
"${PYTHON}" tools/experiments/prepare_frame200_centerpoint.py \
    --source-root "${RAW_ROOT}" --source-info "${RAW_ROOT}/infos_test_200.pkl" \
    --sr-root "${SR_ROOT}" --output-dir "${INFO_ROOT}"

echo "stage=sr46_training status=starting timestamp=$(date --iso-8601=seconds)"
cd "${REPO_ROOT}/tools"
/usr/bin/env PATH="/home/kaixuan-ding/miniconda3/bin:/home/kaixuan-ding/miniconda3/envs/spconv/bin:/usr/local/bin:/usr/bin:/bin" \
    "${PYTHON}" train.py --cfg_file cfgs/radar_models/centerpoint_frame200_350m.yaml \
    --extra_tag frame200_sr46_350m --fix_random_seed --workers 2 --use_amp \
    --logger_iter_interval 20 --structured_log_iter_interval 10 \
    --set DATA_CONFIG.INFO_PATH.train "${INFO_ROOT}/sr_train.pkl" \
    DATA_CONFIG.INFO_PATH.test "${INFO_ROOT}/sr_val.pkl"
echo "stage=sr46_pipeline status=completed timestamp=$(date --iso-8601=seconds)"
