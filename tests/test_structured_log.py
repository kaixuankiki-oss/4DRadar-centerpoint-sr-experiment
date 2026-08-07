import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS_DIR))

from train_utils.structured_log import should_emit_train_metric


class TestStructuredLogSampling(unittest.TestCase):
    def test_emits_first_iteration_after_start_or_resume(self):
        self.assertTrue(should_emit_train_metric(123, 24, 100, 10, is_first_iteration=True))

    def test_emits_at_configured_interval(self):
        self.assertTrue(should_emit_train_metric(120, 20, 100, 10))

    def test_emits_final_iteration_of_epoch(self):
        self.assertTrue(should_emit_train_metric(199, 100, 100, 10))

    def test_skips_unsampled_iteration(self):
        self.assertFalse(should_emit_train_metric(121, 21, 100, 10))

    def test_rejects_non_positive_interval(self):
        with self.assertRaises(ValueError):
            should_emit_train_metric(1, 1, 100, 0)


if __name__ == '__main__':
    unittest.main()
