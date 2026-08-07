#!/usr/bin/env python3
"""Offline score/NMS scan for the trained exp23 checkpoint.

This script runs eval-only jobs with config overrides. It must not be used as a
formal exp29 result unless a follow-up from-scratch training config is approved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from radial_rcs_report_utils import (
    MetricRow,
    format_markdown_table,
    load_metric_row,
    missing_required,
)


DEFAULT_CKPT = (
    "/experiments/a100_radial_wing_multiframe_output/radar_models/"
    "exp23_front5_radial_consistency_gate/"
    "exp23_front5_radial_consistency_gate_4gpu_40ep_20260619/"
    "ckpt/checkpoint_epoch_40.pth"
)

SETTINGS = [
    ("score005_nms020", 0.05, 0.20),
    ("score008_nms020", 0.08, 0.20),
    ("score010_nms020", 0.10, 0.20),
    ("score012_nms020", 0.12, 0.20),
    ("score008_nms015", 0.08, 0.15),
    ("score010_nms015", 0.10, 0.15),
]


def run_eval(repo: Path, root: Path, ckpt: Path, name: str, score: float, nms: float, workers: int) -> tuple[int, Path]:
    log = root / "exp29_offline_threshold_scan" / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "test.py",
        "--cfg_file",
        "cfgs/radar_models/exp23_front5_radial_consistency_gate.yaml",
        "--extra_tag",
        f"exp29_offline_{name}",
        "--ckpt",
        str(ckpt),
        "--batch_size",
        "4",
        "--workers",
        str(workers),
        "--eval_tag",
        name,
        "--set",
        "MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH",
        str(score),
        "MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_THRESH",
        str(nms),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": f"{repo}:{repo / 'tools'}:{os.environ.get('PYTHONPATH', '')}",
        "LD_LIBRARY_PATH": f"/opt/conda/lib/python3.10/site-packages/torch/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }
    with log.open("a", encoding="utf-8") as f:
        f.write("CMD " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=repo / "tools", env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode), log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/workspace/worktrees/a100_radial_rcs_consistency_20260620")
    parser.add_argument("--root", default="/experiments/a100_radial_rcs_consistency_20260620")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo)
    root = Path(args.root)
    ckpt = Path(args.ckpt)
    report = root / "exp29_offline_threshold_scan" / "reports" / "threshold_scan_summary.txt"
    report.parent.mkdir(parents=True, exist_ok=True)

    rows: list[MetricRow] = []
    statuses = []
    for name, score, nms in SETTINGS:
        if args.dry_run:
            statuses.append({"setting": name, "score": score, "nms": nms, "status": "DRY_RUN"})
            continue
        rc, log = run_eval(repo, root, ckpt, name, score, nms, args.workers)
        statuses.append({"setting": name, "score": score, "nms": nms, "status": "DONE" if rc == 0 else "FAILED", "rc": rc, "log": str(log)})
        if rc == 0:
            row = load_metric_row(f"exp29_offline_{name}", log, note=f"score={score}, nms={nms}")
            missing = missing_required(row.metrics)
            if missing:
                statuses[-1]["missing"] = missing
            rows.append(row)

    lines = [
        "# exp29 offline threshold/NMS scan",
        "",
        f"- source checkpoint: `{ckpt}`",
        "- source architecture: exp23 radial consistency",
        "- purpose: diagnosis only; no from-scratch formal exp29 training is launched by this script.",
        "",
        "## Status",
        "",
        "```json",
        json.dumps(statuses, indent=2, sort_keys=True),
        "```",
        "",
    ]
    if rows:
        lines += ["## Metrics", "", format_markdown_table(rows), ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"THRESHOLD_SCAN_REPORT {report}")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
