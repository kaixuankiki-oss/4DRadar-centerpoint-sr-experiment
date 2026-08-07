#!/usr/bin/env python3
import argparse
import datetime
import json
import math
import random
from pathlib import Path


def write_record(stream, marker, payload, timestamp):
    payload = {'timestamp': timestamp.isoformat(timespec='milliseconds'), **payload}
    stream.write('%s INFO %s %s\n' % (
        timestamp.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3], marker,
        json.dumps(payload, ensure_ascii=True, sort_keys=True)
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--iterations', type=int, default=36)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260611)
    now = datetime.datetime.now().astimezone() - datetime.timedelta(minutes=32)

    with args.output.open('w', encoding='utf-8') as stream:
        write_record(stream, 'RUN_META', {
            'run_name': 'HR-4D training log demo',
            'config': 'cfgs/hr4d_models/radar_pointpillar.yaml',
            'output_dir': str(args.output.parent),
            'model': 'PointPillar-HR4D',
            'epochs': args.epochs,
            'batch_size_per_gpu': 4,
            'world_size': 1,
            'amp': True,
            'branch': 'training_log_qiyunlong',
            'mock': True,
        }, now)
        global_iteration = 0
        best_checkpoints = []
        for epoch in range(1, args.epochs + 1):
            epoch_losses = []
            for iteration in range(1, args.iterations + 1):
                global_iteration += 1
                progress = global_iteration / (args.epochs * args.iterations)
                base = 5.6 * math.exp(-3.1 * progress) + 0.42
                loss = max(0.2, base + rng.uniform(-0.17, 0.17))
                epoch_losses.append(loss)
                lr = 0.003 * min(1.0, global_iteration / 55) * (1.0 - 0.82 * progress)
                batch_ms = 212 + 35 * math.exp(-global_iteration / 20) + rng.uniform(-11, 11)
                now += datetime.timedelta(milliseconds=batch_ms)
                write_record(stream, 'TRAIN_METRIC', {
                    'epoch': epoch,
                    'total_epochs': args.epochs,
                    'iteration': iteration,
                    'iterations_per_epoch': args.iterations,
                    'global_iteration': global_iteration,
                    'progress': progress,
                    'loss': loss,
                    'loss_avg': sum(epoch_losses) / len(epoch_losses),
                    'learning_rate': lr,
                    'grad_norm': max(0.15, 8.4 * math.exp(-2.5 * progress) + rng.uniform(-0.4, 0.4)),
                    'data_time_ms': 24 + rng.uniform(-5, 7),
                    'forward_time_ms': 68 + rng.uniform(-5, 6),
                    'backward_time_ms': 93 + rng.uniform(-7, 8),
                    'optimizer_time_ms': 18 + rng.uniform(-2, 3),
                    'batch_time_ms': batch_ms,
                    'elapsed_seconds': global_iteration * batch_ms / 1000,
                    'eta_seconds': (args.epochs * args.iterations - global_iteration) * batch_ms / 1000,
                    'amp_scale': 65536.0,
                    'losses': {
                        'rpn_loss_cls': loss * 0.38,
                        'rpn_loss_loc': loss * 0.51,
                        'rpn_loss_dir': loss * 0.11,
                    },
                    'gpu_memory_allocated_mb': 6720 + rng.uniform(-80, 120),
                    'gpu_memory_reserved_mb': 8192,
                    'gpu_max_memory_allocated_mb': 7015,
                    'gpu_utilization_percent': 88 + rng.uniform(-7, 8),
                    'gpu_memory_used_mb': 8350 + rng.uniform(-60, 90),
                    'gpu_memory_total_mb': 23034,
                    'gpu_temperature_c': 61 + rng.uniform(-2, 3),
                    'gpu_power_w': 67 + rng.uniform(-4, 6),
                }, now)
            write_record(stream, 'EPOCH_METRIC', {
                'epoch': epoch,
                'total_epochs': args.epochs,
                'global_iteration': global_iteration,
                'loss_avg': sum(epoch_losses) / len(epoch_losses),
                'grad_norm_avg': 8.4 * math.exp(-2.5 * epoch / args.epochs),
                'data_time_ms_avg': 25.1,
                'forward_time_ms_avg': 68.4,
                'backward_time_ms_avg': 94.2,
                'optimizer_time_ms_avg': 18.7,
                'batch_time_ms_avg': 214.8,
                'gpu_memory_allocated_mb': 6815,
                'gpu_utilization_percent': 92,
            }, now)
            if epoch % 2 == 0:
                mean_ap = min(0.82, 0.18 + 0.046 * epoch)
                checkpoint_record = {
                    'epoch': epoch,
                    'score_name': 'hr4d/mean_ap',
                    'map': mean_ap,
                    'metrics': {'hr4d/mean_ap': mean_ap},
                }
                write_record(stream, 'EVAL_METRIC', {
                    'epoch': epoch,
                    'samples': 200,
                    'seconds_per_example': 0.031,
                    'average_predicted_objects': 18.4,
                    'metrics': {
                        'hr4d/mean_ap': mean_ap,
                        'hr4d/overall/mean_ap': mean_ap,
                        'hr4d/overall/Vehicle/mean_ap': min(0.88, mean_ap + 0.04),
                        'hr4d/overall/Pedestrian/mean_ap': max(0.0, mean_ap - 0.08),
                        'recall/roi_0.3': min(0.96, 0.46 + 0.043 * epoch),
                        'recall/rcnn_0.5': min(0.91, 0.31 + 0.047 * epoch),
                        'Vehicle_3d/easy_R40': min(0.86, 0.24 + 0.05 * epoch),
                        'Pedestrian_3d/moderate_R40': min(0.68, 0.12 + 0.041 * epoch),
                    },
                }, now)
                write_record(stream, 'CHECKPOINT_METRIC', checkpoint_record, now)
                best_checkpoints.append({
                    'epoch': epoch,
                    'path': 'checkpoint_epoch_%d.pth' % epoch,
                    'score': mean_ap,
                    'score_name': 'hr4d/mean_ap',
                })
                best_checkpoints = sorted(
                    best_checkpoints,
                    key=lambda item: (item['score'], item['epoch']),
                    reverse=True,
                )[:3]
                write_record(stream, 'BEST_CHECKPOINTS', {
                    'top_k': 3,
                    'checkpoints': best_checkpoints,
                }, now)
        if best_checkpoints:
            best = best_checkpoints[0]
            write_record(stream, 'FINAL_BEST_CHECKPOINT_EVAL', {
                'epoch': best['epoch'],
                'checkpoint': best['path'],
                'score_name': best['score_name'],
                'score': best['score'],
                'metrics': {'hr4d/mean_ap': best['score'] + 0.001},
            }, now)
    print(args.output)


if __name__ == '__main__':
    main()
