#!/usr/bin/env python3
"""Shared metric parsing and reporting for A100 radial/RCS experiments."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib import request


METRIC_FIELDS = [
    "pred/frame",
    "overall",
    "0-50m",
    "50-100m",
    "100-150m",
    "150-200m",
    "Car",
    "LargeVehicle",
    "Cyclist",
    "Pedestrian",
]

METRIC_KEYS = {
    "overall": ["hr4d/mean_ap", "hr4d/overall/mean_ap"],
    "0-50m": ["hr4d/0-50m/mean_ap"],
    "50-100m": ["hr4d/50-100m/mean_ap"],
    "100-150m": ["hr4d/100-150m/mean_ap"],
    "150-200m": ["hr4d/150-200m/mean_ap"],
    "Car": ["hr4d/overall/Car/mean_ap"],
    "LargeVehicle": [
        "hr4d/overall/LargeVehicle/mean_ap",
        "hr4d/overall/LargeV/mean_ap",
    ],
    "Cyclist": ["hr4d/overall/Cyclist/mean_ap"],
    "Pedestrian": ["hr4d/overall/Pedestrian/mean_ap"],
}

HISTORICAL_BASELINE = {
    "pred/frame": 33.445,
    "overall": 0.3687,
    "0-50m": 0.5141,
    "50-100m": 0.3467,
    "100-150m": 0.2318,
    "150-200m": 0.1505,
    "Car": 0.7232,
    "LargeVehicle": 0.5333,
    "Cyclist": 0.1782,
    "Pedestrian": 0.0400,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class MetricRow:
    name: str
    metrics: dict[str, float | None]
    note: str = ""


def parse_eval_metric_json(eval_log: Path) -> dict:
    data = {}
    if not eval_log.exists():
        raise FileNotFoundError(eval_log)
    for line in eval_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"EVAL_METRIC\s+(\{.*\})", line)
        if not match:
            continue
        data = json.loads(match.group(1))
    if not data:
        raise ValueError(f"EVAL_METRIC JSON not found in {eval_log}")
    return data


def extract_metrics(eval_metric: dict) -> dict[str, float | None]:
    metrics_blob = eval_metric.get("metrics", {})
    row: dict[str, float | None] = {
        "pred/frame": eval_metric.get("average_predicted_objects")
    }
    for label, keys in METRIC_KEYS.items():
        value = None
        for key in keys:
            if key in metrics_blob:
                value = metrics_blob[key]
                break
        row[label] = value
    return row


def load_metric_row(name: str, eval_log: Path, note: str = "") -> MetricRow:
    return MetricRow(name=name, metrics=extract_metrics(parse_eval_metric_json(eval_log)), note=note)


def missing_required(metrics: dict[str, float | None]) -> list[str]:
    return [field for field in METRIC_FIELDS if metrics.get(field) is None and field != "pred/frame"]


def format_value(value: float | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_delta(value: float | None, ref: float | None) -> str:
    if value is None or ref is None:
        return "NA"
    return f"{value - ref:+.4f}"


def format_markdown_table(rows: list[MetricRow], baseline: dict[str, float | None] | None = None) -> str:
    headers = ["row"] + METRIC_FIELDS
    if baseline is not None:
        headers += ["d_overall_base", "d_150-200_base", "d_pred_base"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = [row.name] + [format_value(row.metrics.get(field)) for field in METRIC_FIELDS]
        if baseline is not None:
            values += [
                format_delta(row.metrics.get("overall"), baseline.get("overall")),
                format_delta(row.metrics.get("150-200m"), baseline.get("150-200m")),
                format_delta(row.metrics.get("pred/frame"), baseline.get("pred/frame")),
            ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def reference_rows() -> list[MetricRow]:
    return [
        MetricRow("A100 single-frame baseline", dict(HISTORICAL_BASELINE), "historical fixed baseline"),
    ]


def format_current_report(
    *,
    current: MetricRow,
    exp20: MetricRow,
    exp23: MetricRow,
    cfg: str,
    tag: str,
    ckpt: Path,
    eval_log: Path,
    branch: str,
    protocol: str,
) -> str:
    rows = reference_rows() + [exp20, exp23, current]
    missing = missing_required(current.metrics)
    if missing:
        raise ValueError(f"{current.name} missing required metrics: {', '.join(missing)}")
    lines = [
        f"# {current.name} final eval",
        "",
        f"- cfg: `{cfg}`",
        f"- tag: `{tag}`",
        f"- checkpoint: `{ckpt}`",
        f"- eval log: `{eval_log}`",
        f"- branch: `{branch}`",
        "- data: `/data/highway_sweeps/highway_train.pkl`, `/data/highway_sweeps/highway_test.pkl`",
        f"- protocol: {protocol}",
        "",
        "## Baseline Comparison",
        "",
        format_markdown_table(rows, baseline=HISTORICAL_BASELINE),
        "",
        "## Reporting Guardrails",
        "",
        "- LargeVehicle is parsed with `LargeVehicle` -> `LargeV` alias fallback.",
        "- Final evaluator remains the fixed HR-4D evaluator; no evaluator or BEV/FOV change is made.",
        "- Radar-aware point spread is handled inside representation/gating, not by widening the main heatmap.",
    ]
    return "\n".join(lines) + "\n"


def format_exp20_24_followup(rows: list[MetricRow]) -> str:
    missing = {row.name: missing_required(row.metrics) for row in rows}
    missing = {name: fields for name, fields in missing.items() if fields}
    if missing:
        raise ValueError(f"missing required metrics: {missing}")
    lines = [
        "# A100 RADAR_FRONT exp20-24 baseline follow-up",
        "",
        "Previous Feishu result messages were sent successfully, but they lacked the required baseline comparison table and LargeVehicle/LargeV AP. This follow-up uses the same eval logs and the fixed parser.",
        "",
        "## Comparison",
        "",
        format_markdown_table(reference_rows() + rows, baseline=HISTORICAL_BASELINE),
        "",
        "## Current Decision",
        "",
        "- exp23 radial consistency remains the main line: it is the only exp20-24 result improving overall, 150-200m, and Car against the historical single-frame baseline.",
        "- exp23 risk is high pred/frame, so exp25-27 focus on lower gate amplitude and RCS/power temporal consistency before any formal new queue is launched.",
        "- exp30/exp31 dual-front ideas are cancelled for this suite; follow-ups use only RADAR_FRONT.",
    ]
    return "\n".join(lines) + "\n"


def source_feishu_env_files(log_path: Path | None = None) -> dict[str, str]:
    env = {}
    for env_file in (
        Path("/root/.codex/feishu_bot.env"),
        Path("/root/.feishu_bot.env"),
        Path("/experiments/feishu_bot.env"),
    ):
        if not env_file.exists():
            continue
        try:
            for raw_line in env_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key in {"FEISHU_BOT_WEBHOOK", "FEISHU_BOT_SECRET"}:
                    env[key] = value
        except OSError as exc:
            if log_path is not None:
                append_jsonl(log_path, {
                    "time": utc_now(),
                    "event": "feishu_env_read_failed",
                    "path": str(env_file),
                    "error": str(exc),
                })
    return env


def feishu_sign(timestamp: str, secret: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(string_to_sign, b"", digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def send_feishu(title: str, text: str, *, log_path: Path, outbox_path: Path) -> bool:
    append_jsonl(outbox_path, {"time": utc_now(), "title": title, "text": text[:4000]})
    env = {**source_feishu_env_files(log_path), **os.environ}
    webhook = env.get("FEISHU_BOT_WEBHOOK", "").strip()
    secret = env.get("FEISHU_BOT_SECRET", "").strip()
    if not webhook:
        append_jsonl(log_path, {
            "time": utc_now(),
            "event": "send_skipped_missing_webhook_env",
            "title": title,
        })
        return False
    payload: dict = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": title,
                    "content": [[{"tag": "text", "text": text[:9000]}]],
                }
            }
        },
    }
    if secret:
        timestamp = str(int(time.time()))
        payload["timestamp"] = timestamp
        payload["sign"] = feishu_sign(timestamp, secret)
    req = request.Request(
        webhook,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            append_jsonl(log_path, {
                "time": utc_now(),
                "event": "send_done",
                "title": title,
                "http_status": resp.status,
                "response": body[:500],
            })
            return resp.status < 400
    except Exception as exc:  # noqa: BLE001
        append_jsonl(log_path, {
            "time": utc_now(),
            "event": "send_failed",
            "title": title,
            "error": str(exc),
        })
        return False
