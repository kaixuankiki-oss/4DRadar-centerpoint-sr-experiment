import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'train_utils' / 'training_output.py'
SPEC = importlib.util.spec_from_file_location('training_output_test_module', MODULE_PATH)
training_output = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training_output)


class TestTrainingOutput(unittest.TestCase):
    def test_explicit_shared_root_namespaces_user_and_container(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {
            'HR4D_TRAINING_OUTPUT_ROOT': directory,
            'HR4D_TRAINING_USER': 'Yunlong Qi',
            'HR4D_TRAINING_CONTAINER': 'qiyunlong/hr4d',
            'HR4D_TRAINING_HOST': 'a100-server',
        }, clear=True):
            output_dir, metadata = training_output.resolve_training_output_dir(
                'output', '/repo', 'radar_models', 'second_radar', 'exp01'
            )

        self.assertEqual(
            output_dir,
            Path(directory) / 'Yunlong-Qi' / 'qiyunlong-hr4d'
            / 'radar_models' / 'second_radar' / 'exp01',
        )
        self.assertTrue(metadata['shared_output'])
        self.assertEqual(metadata['source_user'], 'Yunlong Qi')
        self.assertEqual(metadata['source_container'], 'qiyunlong/hr4d')
        self.assertEqual(metadata['source_host'], 'a100-server')
        self.assertEqual(metadata['training_output_root'], directory)

    def test_canonical_server_root_is_used_when_mounted(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            training_output, 'DEFAULT_SHARED_OUTPUT_ROOT', Path(directory)
        ), mock.patch.dict(os.environ, {
            'USER': 'hr4d',
            'HOSTNAME': 'training-container',
        }, clear=True):
            output_dir, metadata = training_output.resolve_training_output_dir(
                'output', '/repo', 'group', 'model', 'default'
            )

        self.assertEqual(output_dir.parts[-5:], ('hr4d', 'training-container', 'group', 'model', 'default'))
        self.assertTrue(metadata['shared_output'])

    def test_local_output_fallback_preserves_existing_layout(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            training_output, 'DEFAULT_SHARED_OUTPUT_ROOT', Path(directory) / 'missing'
        ), mock.patch.dict(os.environ, {}, clear=True):
            output_dir, metadata = training_output.resolve_training_output_dir(
                'output', directory, 'group', 'model', 'default'
            )

        self.assertEqual(
            output_dir,
            (Path(directory) / 'output' / 'group' / 'model' / 'default').resolve(),
        )
        self.assertFalse(metadata['shared_output'])


if __name__ == '__main__':
    unittest.main()
