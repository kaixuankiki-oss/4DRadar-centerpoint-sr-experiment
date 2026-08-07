#!/usr/bin/env python3
import argparse
import base64
import hmac
import json
import math
import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


MARKERS = (
    'FINAL_BEST_CHECKPOINT_EVAL',
    'CHECKPOINT_METRIC',
    'BEST_CHECKPOINTS',
    'RUN_META',
    'TRAIN_METRIC',
    'EPOCH_METRIC',
    'EVAL_METRIC',
)
MAP_METRIC_PRIORITY = (
    'hr4d/mean_ap',
    'hr4d/overall/mean_ap',
    'mAP',
    'mean_ap',
    'map',
)


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def epoch_value(record):
    number = finite_float(record.get('epoch'))
    if number is None:
        return record.get('epoch')
    return int(number) if number.is_integer() else number


def select_map_metric(metrics):
    numeric_metrics = {}
    for key, value in (metrics or {}).items():
        number = finite_float(value)
        if number is not None:
            numeric_metrics[str(key)] = number
    if not numeric_metrics:
        return None

    lower_to_key = {key.lower(): key for key in numeric_metrics}
    for priority_key in MAP_METRIC_PRIORITY:
        key = lower_to_key.get(priority_key.lower())
        if key is not None:
            return key, numeric_metrics[key]

    for key, value in numeric_metrics.items():
        lower = key.lower()
        if lower.endswith('/mean_ap') and '/overall/' in lower:
            return key, value

    for key, value in numeric_metrics.items():
        parts = key.split('/')
        if parts[-1].lower() == 'mean_ap' and len(parts) <= 3:
            return key, value

    for key, value in numeric_metrics.items():
        if key.lower().endswith('mean_ap'):
            return key, value

    for key, value in numeric_metrics.items():
        if re.search(r'(^|[/_])m?ap($|[/_])', key.lower()):
            return key, value
    return None


def summarize_run(parsed):
    latest_map = None
    best_map = None
    for record in parsed['evaluation']:
        selected = select_map_metric(record.get('metrics'))
        if selected is None:
            continue
        metric, value = selected
        point = {
            'metric': metric,
            'value': value,
            'epoch': epoch_value(record),
            'timestamp': record.get('timestamp'),
        }
        latest_map = point
        if best_map is None or value > best_map['value']:
            best_map = point
    return {
        'best_map': best_map,
        'latest_map': latest_map,
        'evaluation_points': len(parsed['evaluation']),
        'latest_best_checkpoints': parsed['best_checkpoints'][-1] if parsed['best_checkpoints'] else None,
        'final_best_checkpoint': parsed['final_best_checkpoint'],
    }


def evaluation_sort_key(record):
    epoch = finite_float(record.get('epoch'))
    return (epoch is None, epoch if epoch is not None else 0.0, str(record.get('timestamp', '')))


def companion_metrics_log(path):
    if not path.name.startswith('train_') or path.name.startswith('train_metrics_'):
        return None
    suffix = path.name[len('train_'):]
    exact = path.with_name('train_metrics_' + suffix)
    if exact.is_file() and exact.stat().st_size > 0:
        return exact

    candidates = sorted(
        (
            candidate for candidate in path.parent.glob('train_metrics_*.log')
            if candidate.is_file() and candidate.stat().st_size > 0
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if len(candidates) == 1 else None


def run_log_paths(path):
    companion = companion_metrics_log(path)
    return [path, companion] if companion is not None else [path]


def resolve_password(args, parser):
    if args.password and args.password_file:
        parser.error('use either --password or --password-file, not both')
    if args.password:
        password = args.password.strip()
    elif args.password_file:
        password = args.password_file.read_text(encoding='utf-8').strip()
    else:
        parser.error('one of --password or --password-file is required')
    if not password:
        parser.error('password is empty')
    return password


def iter_run_logs(log_root):
    paths = list(log_root.rglob('*.log'))
    directories_with_train_logs = {
        path.parent for path in paths if path.name.startswith('train_') and path.stat().st_size > 0
    }
    paired_metrics_logs = {
        companion for path in paths for companion in [companion_metrics_log(path)]
        if companion is not None
    }
    return sorted(
        (
            path for path in paths
            if not (path.name == 'launcher.log' and path.parent in directories_with_train_logs)
            and path not in paired_metrics_logs
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def parse_log(path):
    result = {
        'meta': {},
        'train': [],
        'epochs': [],
        'evaluation': [],
        'checkpoint_metrics': [],
        'best_checkpoints': [],
        'final_best_checkpoint': None,
        'tail': [],
    }
    lines = []
    paths = run_log_paths(path)
    for log_path in paths:
        with log_path.open('r', encoding='utf-8', errors='replace') as stream:
            lines.extend(stream.readlines())
    result['tail'] = [line.rstrip() for line in lines[-80:]]
    for line in lines:
        for marker in MARKERS:
            token = marker + ' '
            position = line.find(token)
            if position < 0:
                continue
            try:
                payload = json.loads(line[position + len(token):].strip())
            except json.JSONDecodeError:
                break
            if marker == 'RUN_META':
                result['meta'].update(payload)
            elif marker == 'TRAIN_METRIC':
                result['train'].append(payload)
            elif marker == 'EPOCH_METRIC':
                result['epochs'].append(payload)
            elif marker == 'EVAL_METRIC':
                result['evaluation'].append(payload)
            elif marker == 'CHECKPOINT_METRIC':
                result['checkpoint_metrics'].append(payload)
            elif marker == 'BEST_CHECKPOINTS':
                result['best_checkpoints'].append(payload)
            elif marker == 'FINAL_BEST_CHECKPOINT_EVAL':
                result['final_best_checkpoint'] = payload
            break
    result['evaluation'].sort(key=evaluation_sort_key)
    result['checkpoint_metrics'].sort(key=evaluation_sort_key)
    result['summary'] = summarize_run(result)
    result['path'] = str(path)
    result['paths'] = [str(log_path) for log_path in paths]
    result['updated_at'] = max(log_path.stat().st_mtime for log_path in paths)
    return result


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = 'TrainingLogDashboard/1.0'

    def log_message(self, format_string, *args):
        return

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def is_authorized(self):
        authorization = self.headers.get('Authorization', '')
        if not authorization.startswith('Basic '):
            return False
        try:
            decoded = base64.b64decode(authorization[6:], validate=True).decode('utf-8')
            username, password = decoded.split(':', 1)
        except (ValueError, UnicodeDecodeError):
            return False
        return hmac.compare_digest(username, self.server.auth_username) and hmac.compare_digest(
            password, self.server.auth_password
        )

    def request_authorization(self):
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="HR-4D Training Monitor", charset="UTF-8"')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        if not self.is_authorized():
            self.request_authorization()
            return
        request = urlparse(self.path)
        if request.path == '/api/runs':
            runs = []
            for path in iter_run_logs(self.server.log_root):
                try:
                    parsed = parse_log(path)
                except OSError:
                    continue
                if not any((parsed['train'], parsed['epochs'], parsed['evaluation'])):
                    continue
                relative = str(path.relative_to(self.server.log_root)).replace(os.sep, '/')
                runs.append({
                    'id': relative,
                    'name': parsed['meta'].get('run_name') or path.stem,
                    'model': parsed['meta'].get('model', 'unknown'),
                    'source_user': parsed['meta'].get('source_user', 'unknown'),
                    'source_container': parsed['meta'].get('source_container', 'unknown'),
                    'source_host': parsed['meta'].get('source_host', 'unknown'),
                    'updated_at': parsed['updated_at'],
                    'points': len(parsed['train']),
                    'epochs': len(parsed['epochs']),
                    'summary': parsed['summary'],
                })
            return self.send_json({'runs': runs})

        if request.path == '/api/run':
            run_id = parse_qs(request.query).get('id', [''])[0]
            candidate = (self.server.log_root / run_id).resolve()
            try:
                candidate.relative_to(self.server.log_root)
            except ValueError:
                return self.send_json({'error': 'invalid run path'}, 400)
            if not candidate.is_file() or candidate.suffix != '.log':
                return self.send_json({'error': 'run not found'}, 404)
            return self.send_json(parse_log(candidate))

        asset = 'index.html' if request.path == '/' else request.path.lstrip('/')
        candidate = (self.server.static_root / asset).resolve()
        try:
            candidate.relative_to(self.server.static_root)
        except ValueError:
            self.send_error(403)
            return
        if not candidate.is_file():
            self.send_error(404)
            return
        data = candidate.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', mimetypes.guess_type(candidate.name)[0] or 'application/octet-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description='Visualize OpenPCDet structured training logs')
    parser.add_argument('--log-root', type=Path, default=Path('output'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8088)
    parser.add_argument('--username', required=True)
    parser.add_argument('--password', default=None)
    parser.add_argument('--password-file', type=Path, default=None)
    args = parser.parse_args()
    static_root = Path(__file__).resolve().parent / 'static'
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.log_root = args.log_root.resolve()
    server.static_root = static_root.resolve()
    server.auth_username = args.username
    server.auth_password = resolve_password(args, parser)
    print('Training dashboard: http://%s:%d' % (args.host, args.port), flush=True)
    print('Log root: %s' % server.log_root, flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
