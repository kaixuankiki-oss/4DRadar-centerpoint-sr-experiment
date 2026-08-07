import importlib.util
import json
import argparse
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'training_dashboard' / 'server.py'
SPEC = importlib.util.spec_from_file_location('training_dashboard_test_module', MODULE_PATH)
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


class TestTrainingDashboard(unittest.TestCase):
    def test_launcher_log_is_hidden_when_train_log_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / 'run'
            run_dir.mkdir()
            launcher_log = run_dir / 'launcher.log'
            train_log = run_dir / 'train_20260615.log'
            launcher_log.write_text('duplicate output\n', encoding='utf-8')
            train_log.write_text('structured output\n', encoding='utf-8')

            paths = dashboard.iter_run_logs(Path(directory))

        self.assertEqual(paths, [train_log])

    def test_launcher_log_is_kept_without_train_log(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher_log = Path(directory) / 'launcher.log'
            launcher_log.write_text('only output\n', encoding='utf-8')

            paths = dashboard.iter_run_logs(Path(directory))

        self.assertEqual(paths, [launcher_log])

    def test_parse_log_reports_best_and_latest_map(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'train_20260616.log'
            records = [
                ('RUN_META', {'run_name': 'map-demo'}),
                ('EVAL_METRIC', {'epoch': 4, 'metrics': {'hr4d/mean_ap': 0.31}}),
                ('EVAL_METRIC', {'epoch': 8, 'metrics': {'hr4d/mean_ap': 0.47}}),
                ('EVAL_METRIC', {'epoch': 6, 'metrics': {'hr4d/mean_ap': 0.42}}),
            ]
            log_path.write_text(
                '\n'.join('%s %s' % (marker, json.dumps(payload)) for marker, payload in records),
                encoding='utf-8',
            )

            parsed = dashboard.parse_log(log_path)

        self.assertEqual([record['epoch'] for record in parsed['evaluation']], [4, 6, 8])
        self.assertEqual(parsed['summary']['best_map']['epoch'], 8)
        self.assertAlmostEqual(parsed['summary']['best_map']['value'], 0.47)
        self.assertEqual(parsed['summary']['latest_map']['epoch'], 8)

    def test_train_and_metrics_logs_are_paired_as_one_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / 'run'
            run_dir.mkdir()
            train_log = run_dir / 'train_20260622-033243.log'
            metrics_log = run_dir / 'train_metrics_20260622-033243.log'
            train_log.write_text(
                'RUN_META %s\nEVAL_METRIC %s\n' % (
                    json.dumps({'run_name': 'paired-demo'}),
                    json.dumps({'epoch': 1, 'metrics': {'hr4d/mean_ap': 0.3}}),
                ),
                encoding='utf-8',
            )
            metrics_log.write_text(
                'TRAIN_METRIC %s\nEPOCH_METRIC %s\n' % (
                    json.dumps({'epoch': 1, 'global_iteration': 10, 'loss': 1.2}),
                    json.dumps({'epoch': 1, 'loss_avg': 1.1}),
                ),
                encoding='utf-8',
            )

            paths = dashboard.iter_run_logs(Path(directory))
            parsed = dashboard.parse_log(train_log)

        self.assertEqual(paths, [train_log])
        self.assertEqual(parsed['meta']['run_name'], 'paired-demo')
        self.assertEqual(parsed['train'][0]['loss'], 1.2)
        self.assertEqual(parsed['epochs'][0]['loss_avg'], 1.1)
        self.assertEqual(parsed['summary']['best_map']['value'], 0.3)
        self.assertEqual(parsed['paths'], [str(train_log), str(metrics_log)])

    def test_parse_log_reports_checkpoint_records(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'train_metrics_20260622.log'
            records = [
                ('CHECKPOINT_METRIC', {
                    'epoch': 2, 'score_name': 'hr4d/mean_ap', 'map': 0.22,
                    'metrics': {'hr4d/mean_ap': 0.22},
                }),
                ('BEST_CHECKPOINTS', {
                    'top_k': 3,
                    'checkpoints': [
                        {'epoch': 2, 'path': 'checkpoint_epoch_2.pth', 'score': 0.22,
                         'score_name': 'hr4d/mean_ap'},
                    ],
                }),
                ('FINAL_BEST_CHECKPOINT_EVAL', {
                    'epoch': 2,
                    'checkpoint': '/tmp/checkpoint_epoch_2.pth',
                    'score_name': 'hr4d/mean_ap',
                    'score': 0.22,
                    'metrics': {'hr4d/mean_ap': 0.221},
                }),
            ]
            log_path.write_text(
                '\n'.join('%s %s' % (marker, json.dumps(payload)) for marker, payload in records),
                encoding='utf-8',
            )

            parsed = dashboard.parse_log(log_path)

        self.assertEqual(parsed['checkpoint_metrics'][0]['epoch'], 2)
        self.assertEqual(parsed['best_checkpoints'][0]['checkpoints'][0]['path'], 'checkpoint_epoch_2.pth')
        self.assertEqual(parsed['summary']['latest_best_checkpoints']['top_k'], 3)
        self.assertEqual(parsed['final_best_checkpoint']['checkpoint'], '/tmp/checkpoint_epoch_2.pth')
        self.assertEqual(parsed['summary']['final_best_checkpoint']['epoch'], 2)

    def test_resolve_password_accepts_inline_password(self):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(password='hr4d', password_file=None)

        self.assertEqual(dashboard.resolve_password(args, parser), 'hr4d')

    def test_resolve_password_accepts_password_file(self):
        with tempfile.TemporaryDirectory() as directory:
            password_file = Path(directory) / 'password.txt'
            password_file.write_text('secret\n', encoding='utf-8')
            parser = argparse.ArgumentParser()
            args = argparse.Namespace(password=None, password_file=password_file)

            self.assertEqual(dashboard.resolve_password(args, parser), 'secret')

    def test_resolve_password_rejects_both_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            password_file = Path(directory) / 'password.txt'
            password_file.write_text('secret\n', encoding='utf-8')
            parser = argparse.ArgumentParser()
            args = argparse.Namespace(password='hr4d', password_file=password_file)

            with self.assertRaises(SystemExit):
                dashboard.resolve_password(args, parser)

    def test_select_map_metric_falls_back_to_overall_mean_ap(self):
        selected = dashboard.select_map_metric({
            'recall/roi_0.3': 0.9,
            'hr4d/0-50m/Vehicle/mean_ap': 0.2,
            'hr4d/overall/mean_ap': 0.4,
        })

        self.assertEqual(selected, ('hr4d/overall/mean_ap', 0.4))

    def test_parse_log_tail_comes_from_selected_file(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'train_20260616.log'
            log_path.write_text(
                '\n'.join('line %03d' % index for index in range(100)) + '\nselected tail',
                encoding='utf-8',
            )

            parsed = dashboard.parse_log(log_path)

        self.assertEqual(parsed['tail'][-1], 'selected tail')


if __name__ == '__main__':
    unittest.main()
