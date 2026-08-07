import unittest

import torch
from easydict import EasyDict

from pcdet.models.backbones_3d.vfe.radar_pillar_vfe import RadarPillarVFE


class TestRadarPillarVFE(unittest.TestCase):
    def test_rcs_statistics_ignore_padding(self):
        config = EasyDict({
            'STAT_FEATURE_INDICES': [3],
            'INCLUDE_POINT_COUNT': True,
            'MAX_POINTS_PER_VOXEL': 4,
        })
        vfe = RadarPillarVFE(config, num_point_features=5)
        voxels = torch.tensor([
            [[1.0, 2.0, 3.0, 2.0, 10.0],
             [3.0, 4.0, 5.0, 6.0, 20.0],
             [0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]],
            [[2.0, 1.0, 0.0, 5.0, 7.0],
             [99.0, 99.0, 99.0, 99.0, 99.0],
             [0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]],
        ])
        output = vfe({
            'voxels': voxels,
            'voxel_num_points': torch.tensor([2, 1]),
        })['voxel_features']

        self.assertEqual(vfe.get_output_feature_dim(), 8)
        self.assertEqual(tuple(output.shape), (2, 8))
        torch.testing.assert_close(
            output[0],
            torch.tensor([2.0, 3.0, 4.0, 4.0, 15.0, 6.0, 2.0, 0.5]),
        )
        torch.testing.assert_close(
            output[1],
            torch.tensor([2.0, 1.0, 0.0, 5.0, 7.0, 5.0, 0.0, 0.25]),
        )

    def test_invalid_stat_index_is_rejected(self):
        config = EasyDict({'STAT_FEATURE_INDICES': [5]})
        with self.assertRaises(ValueError):
            RadarPillarVFE(config, num_point_features=5)


if __name__ == '__main__':
    unittest.main()
