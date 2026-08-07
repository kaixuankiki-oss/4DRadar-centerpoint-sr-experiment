# Training Log Dashboard

This dashboard reads structured records embedded in normal OpenPCDet `.log` files. It does not use TensorBoard.

```bash
python tools/training_dashboard/server.py --log-root output --host 127.0.0.1 --port 8088 \
  --username hr4d-training --password hr4d
```

Open the dashboard with:

```text
http://hr4d-training:hr4d@127.0.0.1:8088/
```

For a Linux training server or offline intranet machine, run it from the repository root and point `--log-root`
to the directory that contains training output folders:

```bash
cd /path/to/4DRadar
python tools/training_dashboard/server.py \
  --log-root /mnt/nvme-ai-data/4DRadar/output \
  --host 0.0.0.0 \
  --port 8088 \
  --username hr4d-training \
  --password hr4d
```

Then open:

```text
http://hr4d-training:hr4d@<server-ip>:8088/
```

If the server port is not directly reachable, use SSH port forwarding from your local machine:

```bash
ssh -L 8088:127.0.0.1:8088 <user>@<server-ip>
```

and start the server with `--host 127.0.0.1`, then open:

```text
http://hr4d-training:hr4d@127.0.0.1:8088/
```

The dashboard is self-contained and does not require internet access after the repository and Python environment
are already available on the server.

Supported records are `RUN_META`, `TRAIN_METRIC`, `EPOCH_METRIC`, `EVAL_METRIC`, `CHECKPOINT_METRIC`,
`BEST_CHECKPOINTS`, and `FINAL_BEST_CHECKPOINT_EVAL`, followed by one JSON object.

When a run directory contains both `train_<timestamp>.log` and `train_metrics_<timestamp>.log`, the dashboard
automatically treats them as one run. The main log contributes configuration and evaluation text, while the
metrics log contributes loss curves, structured evaluation records, and checkpoint summaries.

You can also point `--log-root` at a single experiment output directory, for example:

```bash
python tools/training_dashboard/server.py \
  --log-root /mnt/nvme-ai-data/4DRadar/output/root/container/radar_models/centerpoint_highway_0_200m_split_minpts3/run_tag \
  --host 127.0.0.1 \
  --port 8088 \
  --username hr4d-training \
  --password hr4d
```

Use `--password <value>` for quick offline use. If you prefer not to expose the password in the process command,
use `--password-file /path/to/password` instead.

`TRAIN_METRIC` is sampled every 10 iterations by default, while the first metric after starting or resuming and
the final metric of every epoch are always recorded. Override the sampling interval when starting training:

```bash
python tools/train.py ... --structured_log_iter_interval 20
```

For a 100,000-frame, 40-epoch run with batch size 16, the default interval reduces structured iteration records
from roughly 250,000 to 25,000. This targets about one tenth of the previous structured log volume.

## Server-wide training logs

On a shared training server, mount one host directory into every training container and set these variables:

```bash
HR4D_TRAINING_OUTPUT_ROOT=/workspace/4DRadar/output
HR4D_TRAINING_USER=<developer-name>
HR4D_TRAINING_CONTAINER=<container-name>
HR4D_TRAINING_HOST=<server-name>
```

`tools/train.py` stores each run under:

```text
${HR4D_TRAINING_OUTPUT_ROOT}/<user>/<container>/<config-group>/<config>/<extra-tag>/
```

The dashboard scans the shared root recursively, so runs from different accounts and containers appear in the same run selector without overwriting each other. On the HR-4D A100 server, the canonical host directory is:

```text
/mnt/nvme-ai-data/4DRadar/output
```

If `HR4D_TRAINING_OUTPUT_ROOT` is not set, the training code automatically uses that canonical path when it is mounted in the current environment. Otherwise it falls back to the repository's configured `OUTPUT_ROOT` for local development.
