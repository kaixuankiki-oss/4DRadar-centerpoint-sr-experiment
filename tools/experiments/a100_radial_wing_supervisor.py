#!/usr/bin/env python3
"""Durable serial supervisor for A100 radial-wing experiments.

This is intentionally small and explicit: every child job appends logs and
updates a JSON state file. A child failure marks the experiment failed and the
supervisor keeps its own state readable instead of disappearing silently.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import hmac
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib import request


EXPERIMENTS = [
    ("exp20_front5_naive_stack", "cfgs/radar_models/exp20_front5_naive_stack.yaml"),
    ("exp21_front5_range_time_gate_vfe", "cfgs/radar_models/exp21_front5_range_time_gate_vfe.yaml"),
    ("exp22_front5_wing_aux_gate", "cfgs/radar_models/exp22_front5_wing_aux_gate.yaml"),
    ("exp23_front5_radial_consistency_gate", "cfgs/radar_models/exp23_front5_radial_consistency_gate.yaml"),
    ("exp24_front5_combined_candidate", "cfgs/radar_models/exp24_front5_combined_candidate.yaml"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Supervisor:
    repo: Path
    root: Path
    output_root: Path
    gpu_set: str
    workers: int
    batch_size: int
    state_path: Path
    heartbeat_path: Path
    max_retries: int

    @property
    def feishu_log_path(self) -> Path:
        return self.root / "logs" / "feishu.log"

    @property
    def outbox_path(self) -> Path:
        return self.root / "notifications" / "outbox.jsonl"

    def write_state(self, state: dict) -> None:
        state["updated_at"] = utc_now()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def heartbeat(self, state: str) -> None:
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.write_text(
            json.dumps({"time": utc_now(), "state": state}, indent=2),
            encoding="utf-8",
        )

    def append_jsonl(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def source_env_files(self) -> dict[str, str]:
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
                self.append_jsonl(self.feishu_log_path, {
                    "time": utc_now(),
                    "event": "feishu_env_read_failed",
                    "path": str(env_file),
                    "error": str(exc),
                })
        return env

    @staticmethod
    def feishu_sign(timestamp: str, secret: str) -> str:
        string_to_sign = f"{timestamp}\n{secret}".encode("utf-8")
        digest = hmac.new(string_to_sign, b"", digestmod=hashlib.sha256).digest()
        return base64.b64encode(digest).decode("utf-8")

    def send_feishu(self, title: str, text: str) -> bool:
        event = {"time": utc_now(), "title": title, "text": text[:4000]}
        self.append_jsonl(self.outbox_path, event)
        env = {**self.source_env_files(), **os.environ}
        webhook = env.get("FEISHU_BOT_WEBHOOK", "").strip()
        secret = env.get("FEISHU_BOT_SECRET", "").strip()
        if not webhook:
            self.append_jsonl(self.feishu_log_path, {
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
            payload["sign"] = self.feishu_sign(timestamp, secret)
        req = request.Request(
            webhook,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=20) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                self.append_jsonl(self.feishu_log_path, {
                    "time": utc_now(),
                    "event": "send_done",
                    "title": title,
                    "http_status": resp.status,
                    "response": body[:500],
                })
                return resp.status < 400
        except Exception as exc:  # noqa: BLE001 - notification must never crash supervisor.
            self.append_jsonl(self.feishu_log_path, {
                "time": utc_now(),
                "event": "send_failed",
                "title": title,
                "error": str(exc),
            })
            return False

    def run_child(self, name: str, step: str, cmd: list[str], log_path: Path, env: dict | None = None) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {utc_now()} START {name} {step} =====\n")
            log.write("CMD " + " ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=self.repo / "tools",
                env={**os.environ, **(env or {})},
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while proc.poll() is None:
                self.heartbeat(f"{name}:{step}:RUNNING")
                time.sleep(60)
            log.write(f"===== {utc_now()} END {name} {step} rc={proc.returncode} =====\n")
        return int(proc.returncode)

    def run_child_with_retry(
        self, name: str, step: str, cmd: list[str], log_path: Path, env: dict | None = None
    ) -> int:
        rc = 1
        for attempt in range(self.max_retries + 1):
            attempt_log = log_path if attempt == 0 else log_path.with_name(f"{log_path.stem}.retry{attempt}{log_path.suffix}")
            rc = self.run_child(name, step, cmd, attempt_log, env=env)
            if rc == 0:
                return 0
        return rc

    def find_ckpt(self, cfg_stem: str, tag: str) -> Path | None:
        ckpt_dir = self.output_root / "radar_models" / cfg_stem / tag / "ckpt"
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
        return ckpts[-1] if ckpts else None

    def metric_summary(self, eval_log: Path, report_path: Path, name: str, cfg: str, tag: str, ckpt: Path) -> str:
        metrics = {}
        avg_pred = None
        if eval_log.exists():
            for line in eval_log.read_text(encoding="utf-8", errors="replace").splitlines():
                match = re.search(r"EVAL_METRIC\s+(\{.*\})", line)
                if match:
                    data = json.loads(match.group(1))
                    metrics = data.get("metrics", {})
                    avg_pred = data.get("average_predicted_objects")
        keys = [
            ("overall", "hr4d/mean_ap"),
            ("0-50m", "hr4d/0-50m/mean_ap"),
            ("50-100m", "hr4d/50-100m/mean_ap"),
            ("100-150m", "hr4d/100-150m/mean_ap"),
            ("150-200m", "hr4d/150-200m/mean_ap"),
            ("Car", "hr4d/overall/Car/mean_ap"),
            ("LargeVehicle", "hr4d/overall/LargeVehicle/mean_ap"),
            ("Cyclist", "hr4d/overall/Cyclist/mean_ap"),
            ("Pedestrian", "hr4d/overall/Pedestrian/mean_ap"),
        ]
        lines = [
            f"# {name} final eval",
            "",
            f"- cfg: `{cfg}`",
            f"- tag: `{tag}`",
            f"- checkpoint: `{ckpt}`",
            f"- eval log: `{eval_log}`",
            f"- branch: `{os.environ.get('GIT_BRANCH', 'unknown')}`",
            "- data: `/data/highway_sweeps/highway_train.pkl`, `/data/highway_sweeps/highway_test.pkl`",
            "- protocol: RADAR_FRONT, MAX_SWEEPS=5, MIN_GT_POINTS=3, HR-4D evaluator, frozen 0-200m BEV/FOV",
            "",
            "## Metrics",
        ]
        if avg_pred is not None:
            lines.append(f"- average predicted objects: {avg_pred:.3f}")
        for label, key in keys:
            if key in metrics:
                lines.append(f"- {label}: {metrics[key]:.4f}")
        if not metrics:
            lines.append("- EVAL_METRIC JSON not found in eval log; inspect the log manually.")
        text = "\n".join(lines) + "\n"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")
        return text

    def run(self) -> int:
        state = {
            "queue_status": "RUNNING",
            "root": str(self.root),
            "repo": str(self.repo),
            "output_root": str(self.output_root),
            "gpu_set": self.gpu_set,
            "experiments": {name: {"status": "PENDING", "cfg": cfg} for name, cfg in EXPERIMENTS},
        }
        self.write_state(state)
        common_env = {
            "CUDA_VISIBLE_DEVICES": self.gpu_set,
            "PYTHONPATH": f"{self.repo}:{self.repo / 'tools'}:{os.environ.get('PYTHONPATH', '')}",
            "LD_LIBRARY_PATH": f"/opt/conda/lib/python3.10/site-packages/torch/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
            "GIT_BRANCH": os.environ.get("GIT_BRANCH", "codex/a100-radial-wing-multiframe-20260619"),
        }

        for name, cfg in EXPERIMENTS:
            exp_state = state["experiments"][name]
            cfg_stem = Path(cfg).stem
            tag = f"{name}_4gpu_40ep_20260619"
            run_root = self.root / name
            logs = run_root / "logs"
            exp_state.update({"status": "TRAINING", "tag": tag, "run_root": str(run_root)})
            self.write_state(state)
            train_cmd = [
                "torchrun",
                "--standalone",
                "--nproc_per_node=4",
                "train.py",
                "--launcher",
                "pytorch",
                "--cfg_file",
                cfg,
                "--extra_tag",
                tag,
                "--batch_size",
                str(self.batch_size),
                "--workers",
                str(self.workers),
                "--logger_iter_interval",
                "50",
                "--structured_log_iter_interval",
                "10",
                "--ckpt_save_interval",
                "1",
                "--fix_random_seed",
                "--wo_gpu_stat",
            ]
            rc = self.run_child_with_retry(name, "train", train_cmd, logs / "train.log", env=common_env)
            if rc != 0:
                exp_state.update({"status": "TRAIN_FAILED", "train_rc": rc})
                state["queue_status"] = "PAUSED_AFTER_FAILURE"
                self.write_state(state)
                self.send_feishu(
                    f"A100 radial-wing {name} train failed",
                    f"{name} train failed with rc={rc}.\nlog: {logs / 'train.log'}\nstate: {self.state_path}",
                )
                return rc

            ckpt = self.find_ckpt(cfg_stem, tag)
            if ckpt is None:
                exp_state.update({"status": "NO_CKPT"})
                state["queue_status"] = "PAUSED_AFTER_FAILURE"
                self.write_state(state)
                self.send_feishu(
                    f"A100 radial-wing {name} missing checkpoint",
                    f"{name} finished train command but no checkpoint was found.\nstate: {self.state_path}",
                )
                return 2

            exp_state.update({"status": "EVALUATING", "ckpt": str(ckpt)})
            self.write_state(state)
            eval_cmd = [
                "python",
                "test.py",
                "--cfg_file",
                cfg,
                "--extra_tag",
                tag,
                "--ckpt",
                str(ckpt),
                "--batch_size",
                "4",
                "--workers",
                str(self.workers),
                "--eval_tag",
                "final",
            ]
            eval_log = logs / "eval.log"
            rc = self.run_child_with_retry(name, "eval", eval_cmd, eval_log, env=common_env)
            if rc != 0:
                exp_state.update({"status": "EVAL_FAILED", "eval_rc": rc})
                state["queue_status"] = "PAUSED_AFTER_FAILURE"
                self.write_state(state)
                self.send_feishu(
                    f"A100 radial-wing {name} eval failed",
                    f"{name} eval failed with rc={rc}.\nlog: {eval_log}\nstate: {self.state_path}",
                )
                return rc

            report_path = run_root / "reports" / "final_metric_summary.txt"
            summary = self.metric_summary(eval_log, report_path, name, cfg, tag, ckpt)
            exp_state.update({"status": "EVAL_DONE", "report": str(report_path)})
            self.write_state(state)
            self.send_feishu(f"A100 radial-wing {name} eval done", summary)

        state["queue_status"] = "DONE"
        self.write_state(state)
        self.heartbeat("DONE")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/workspace/worktrees/a100_radial_wing_multiframe_20260619")
    parser.add_argument("--root", default="/experiments/a100_radial_wing_multiframe_20260619")
    parser.add_argument("--output-root", default="/experiments/a100_radial_wing_multiframe_output")
    parser.add_argument("--gpu-set", default="4,5,6,7")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if args.status:
        state_path = root / "state" / "queue_state.json"
        heartbeat_path = root / "state" / "heartbeat.json"
        print(state_path.read_text(encoding="utf-8") if state_path.exists() else "{}")
        if heartbeat_path.exists():
            print("\nHEARTBEAT")
            print(heartbeat_path.read_text(encoding="utf-8"))
        return

    lock_path = root / "state" / "supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another supervisor holds lock: {lock_path}")

    sup = Supervisor(
        repo=Path(args.repo),
        root=root,
        output_root=Path(args.output_root),
        gpu_set=args.gpu_set,
        workers=args.workers,
        batch_size=args.batch_size,
        state_path=root / "state" / "queue_state.json",
        heartbeat_path=root / "state" / "heartbeat.json",
        max_retries=args.max_retries,
    )
    raise SystemExit(sup.run())


if __name__ == "__main__":
    main()
