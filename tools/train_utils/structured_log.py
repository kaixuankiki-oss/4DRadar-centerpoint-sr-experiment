import datetime
import json
import math
import subprocess

import torch


def should_emit_train_metric(global_iteration, iteration, iterations_per_epoch, interval, is_first_iteration=False):
    if interval <= 0:
        raise ValueError('structured log interval must be greater than zero')
    return (
        is_first_iteration
        or iteration == iterations_per_epoch
        or global_iteration % interval == 0
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def emit_metric(logger, marker, payload):
    record = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec='milliseconds'),
        **payload,
    }
    logger.info('%s %s', marker, json.dumps(_json_safe(record), ensure_ascii=True, sort_keys=True, allow_nan=False))


def gpu_snapshot(include_utilization=False):
    snapshot = {
        'gpu_memory_allocated_mb': 0.0,
        'gpu_memory_reserved_mb': 0.0,
        'gpu_max_memory_allocated_mb': 0.0,
    }
    if torch.cuda.is_available():
        snapshot.update({
            'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024 ** 2,
            'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 ** 2,
            'gpu_max_memory_allocated_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
        })

    if include_utilization:
        try:
            output = subprocess.check_output([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits', '-i', str(torch.cuda.current_device())
            ], text=True, timeout=2).strip().splitlines()[0]
            utilization, used, total, temperature, power = [float(value.strip()) for value in output.split(',')]
            snapshot.update({
                'gpu_utilization_percent': utilization,
                'gpu_memory_used_mb': used,
                'gpu_memory_total_mb': total,
                'gpu_temperature_c': temperature,
                'gpu_power_w': power,
            })
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    return snapshot
