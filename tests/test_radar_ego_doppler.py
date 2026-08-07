import unittest

import numpy as np

from pcdet.datasets.radar.radar_dataset import RadarDataset


class TestRadarEgoDoppler(unittest.TestCase):
    def test_forward_motion_is_removed_from_front_target(self):
        result = RadarDataset._compensate_ego_doppler(
            np.array([-10.0], dtype=np.float32),
            np.array([[20.0, 0.0, 0.0]], dtype=np.float32),
            {'vehiclespeed': 36.0, 'vehicledirection': 'FORWARD'},
            np.eye(4, dtype=np.float32),
        )
        np.testing.assert_allclose(result, [0.0], atol=1e-5)

    def test_side_target_has_no_translation_compensation(self):
        result = RadarDataset._compensate_ego_doppler(
            np.array([2.5], dtype=np.float32),
            np.array([[0.0, 15.0, 0.0]], dtype=np.float32),
            {'vehiclespeed': 36.0, 'vehicledirection': 'FORWARD'},
            np.eye(4, dtype=np.float32),
        )
        np.testing.assert_allclose(result, [2.5], atol=1e-5)

    def test_reverse_motion_and_sweep_rotation(self):
        body_to_current = np.eye(4, dtype=np.float32)
        body_to_current[:2, :2] = np.array(
            [[0.0, -1.0], [1.0, 0.0]], dtype=np.float32
        )
        result = RadarDataset._compensate_ego_doppler(
            np.array([10.0], dtype=np.float32),
            np.array([[0.0, 20.0, 0.0]], dtype=np.float32),
            {'vehiclespeed': 36.0, 'vehicledirection': 'REVERSE'},
            body_to_current,
        )
        np.testing.assert_allclose(result, [0.0], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
