#!/usr/bin/env python3
"""Generate or send the exp20-24 baseline comparison follow-up."""

from __future__ import annotations

import argparse
from pathlib import Path

from radial_rcs_report_utils import (
    format_exp20_24_followup,
    load_metric_row,
    send_feishu,
)


EXP20_24 = [
    ("exp20_front5_naive_stack", "naive 5-sweep"),
    ("exp21_front5_range_time_gate_vfe", "range/time VFE gate"),
    ("exp22_front5_wing_aux_gate", "wing gate"),
    ("exp23_front5_radial_consistency_gate", "radial consistency"),
    ("exp24_front5_combined_candidate", "radial + wing combined"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/experiments/a100_radial_wing_multiframe_20260619")
    parser.add_argument("--report", default=None)
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for name, note in EXP20_24:
        eval_log = root / name / "logs" / "eval.log"
        rows.append(load_metric_row(name, eval_log, note=note))

    text = format_exp20_24_followup(rows)
    report = Path(args.report) if args.report else root / "reports" / "exp20_24_baseline_followup_summary.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    print(f"FOLLOWUP_REPORT {report}")
    print(text)

    if args.send:
        ok = send_feishu(
            "A100 RADAR_FRONT exp20-24 baseline follow-up",
            text,
            log_path=root / "logs" / "feishu.log",
            outbox_path=root / "notifications" / "outbox.jsonl",
        )
        print(f"FEISHU_SENT {ok}")


if __name__ == "__main__":
    main()
