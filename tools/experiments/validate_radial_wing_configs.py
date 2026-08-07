#!/usr/bin/env python3
"""Validate A100 radial-wing multiframe experiment configs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402


EXPECTED_RANGE = [0.0, -20.0, -8.0, 200.0, 20.0, 8.0]
EXPECTED_PKLS = {"highway_train.pkl", "highway_test.pkl"}
REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"


def reset_cfg():
    cfg.clear()
    cfg.ROOT_DIR = REPO_ROOT
    cfg.LOCAL_RANK = 0


def validate_one(cfg_file: Path) -> None:
    reset_cfg()
    if not cfg_file.is_absolute():
        cfg_file = (REPO_ROOT / cfg_file).resolve()
    cfg_from_yaml_file(str(cfg_file), cfg)
    data_cfg = cfg.DATA_CONFIG
    errors = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    check(data_cfg.DATASET == "RadarDataset", "DATA_CONFIG.DATASET must be RadarDataset")
    check(str(data_cfg.DATA_PATH) == "/data/highway_sweeps", "DATA_PATH must be /data/highway_sweeps")
    check(str(data_cfg.PKL_PATH) == "/data/highway_sweeps", "PKL_PATH must be /data/highway_sweeps")
    info_paths = set(map(str, data_cfg.INFO_PATH.train + data_cfg.INFO_PATH.test))
    check(EXPECTED_PKLS.issubset(info_paths), "must use highway_train.pkl/highway_test.pkl")
    check("train_data.pkl" not in info_paths and "test_data.pkl" not in info_paths, "must not use old single-frame pkl files")
    check(data_cfg.RADAR_SENSOR == "RADAR_FRONT", "RADAR_SENSOR must be RADAR_FRONT")
    check(data_cfg.NUM_POINTS_SENSOR == "RADAR_FRONT", "NUM_POINTS_SENSOR must be RADAR_FRONT")
    check("RADAR_FRONT_TOP" not in str(data_cfg), "RADAR_FRONT_TOP must not be used in this FRONT-only suite")
    check(int(data_cfg.MAX_SWEEPS) == 5, "MAX_SWEEPS must be 5")
    check(int(data_cfg.MIN_GT_POINTS) == 3, "MIN_GT_POINTS must be 3")
    check(list(map(float, data_cfg.POINT_CLOUD_RANGE)) == EXPECTED_RANGE, "POINT_CLOUD_RANGE changed")
    check(list(map(float, cfg.MODEL.DENSE_HEAD.POST_PROCESSING.POST_CENTER_LIMIT_RANGE)) == EXPECTED_RANGE, "POST_CENTER_LIMIT_RANGE changed")
    eval_metric = cfg.MODEL.POST_PROCESSING.EVAL_METRIC
    if isinstance(eval_metric, str):
        eval_metric_name = eval_metric
    else:
        eval_metric_name = eval_metric.get("NAME", None)
    check(eval_metric_name == "hr4d", "EVAL_METRIC must be hr4d")
    check(int(cfg.OPTIMIZATION.NUM_EPOCHS) == 40, "NUM_EPOCHS must be 40")
    check("timestamp" in data_cfg.POINT_FEATURE_ENCODING.used_feature_list, "timestamp feature missing")
    check("doppler" in data_cfg.POINT_FEATURE_ENCODING.used_feature_list, "doppler feature missing")
    amplitude_features = {"RCS", "power"} & set(data_cfg.POINT_FEATURE_ENCODING.used_feature_list)
    check(bool(amplitude_features), "RCS or power feature missing")
    check("Vx" in data_cfg.POINT_FEATURE_ENCODING.used_feature_list, "Vx feature missing")
    check("Vy" in data_cfg.POINT_FEATURE_ENCODING.used_feature_list, "Vy feature missing")

    if errors:
        print(f"CONFIG_INVALID {cfg_file}")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)
    print(
        "CONFIG_OK",
        cfg_file,
        "VFE=" + cfg.MODEL.VFE.NAME,
        "MAP_TO_BEV=" + cfg.MODEL.MAP_TO_BEV.NAME,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_files", nargs="+")
    args = parser.parse_args()
    os.chdir(TOOLS_DIR)
    for name in args.cfg_files:
        validate_one(Path(name))


if __name__ == "__main__":
    main()
