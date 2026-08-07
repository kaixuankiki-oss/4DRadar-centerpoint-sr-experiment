#!/usr/bin/env python3
"""Run a one-batch dataloader/network/loss/backward smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, model_fn_decorator  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])
    log_file = Path(args.log) if args.log else Path("/tmp") / f"{cfg.TAG}_smoke.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(log_file, rank=0)
    logger.info("SMOKE_CFG %s", args.cfg_file)

    train_set, train_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=True,
        seed=666,
    )
    batch = next(iter(train_loader))
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    # Avoid NMS during training smoke; this validates loss/backward only.
    if hasattr(model, "dense_head"):
        model.dense_head.predict_boxes_when_training = False
    model.cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model_func = model_fn_decorator()
    out = model_func(model, batch)
    out.loss.backward()
    optimizer.step()
    print(
        "SMOKE_OK",
        f"cfg={args.cfg_file}",
        f"samples={len(train_set)}",
        f"loss={float(out.loss.detach().cpu()):.6f}",
    )
    for key, value in sorted(out.tb_dict.items()):
        print(f"TB {key}={value}")


if __name__ == "__main__":
    main()
