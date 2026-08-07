import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


class WingAwarePointPillarScatter(PointPillarScatter):
    """PointPillar scatter with a conservative object-wing BEV gate.

    This module keeps the BEV grid and CenterHead targets unchanged. It derives
    per-pillar radar statistics from the voxel points and uses them to gate the
    scattered pillar features.
    """

    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__(model_cfg=model_cfg, grid_size=grid_size, **kwargs)
        hidden_channels = int(self.model_cfg.get('WING_GATE_CHANNELS', 16))
        self.gate_scale = float(self.model_cfg.get('WING_GATE_SCALE', 0.5))
        self.gate_net = nn.Sequential(
            nn.Conv2d(5, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.num_bev_features, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    @staticmethod
    def _safe_mean(values, mask, eps=1e-3):
        denom = mask.sum(dim=1, keepdim=True).clamp_min(eps)
        return (values * mask).sum(dim=1, keepdim=True) / denom

    def _pillar_radar_stats(self, batch_dict):
        voxels = batch_dict['voxels']
        num_points = batch_dict['voxel_num_points']
        point_mask = (
            torch.arange(voxels.shape[1], device=voxels.device)[None, :]
            < num_points[:, None]
        ).type_as(voxels).unsqueeze(-1)

        xy_range = torch.norm(voxels[:, :, 0:2], dim=-1, keepdim=True) / 200.0
        rcs = voxels[:, :, 3:4]
        doppler = voxels[:, :, 5:6]
        timestamp = voxels[:, :, 8:9]
        count = num_points.to(voxels.dtype).view(-1, 1) / float(voxels.shape[1])
        stats = [
            self._safe_mean(xy_range, point_mask),
            self._safe_mean(rcs, point_mask),
            self._safe_mean(doppler.abs(), point_mask),
            self._safe_mean(timestamp, point_mask),
            count.unsqueeze(-1),
        ]
        return torch.cat(stats, dim=-1).squeeze(1)

    def forward(self, batch_dict, **kwargs):
        batch_dict = super().forward(batch_dict, **kwargs)
        pillar_stats = self._pillar_radar_stats(batch_dict)
        coords = batch_dict['voxel_coords']
        batch_size = batch_dict['spatial_features'].shape[0]
        stat_features = torch.zeros(
            batch_size, 5, self.nz * self.nx * self.ny,
            dtype=pillar_stats.dtype, device=pillar_stats.device
        )
        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            stat_features[batch_idx, :, indices.long()] = pillar_stats[batch_mask].t()

        stat_features = stat_features.view(batch_size, 5, self.ny, self.nx)
        gate = self.gate_net(stat_features)
        batch_dict['spatial_features'] = batch_dict['spatial_features'] * (
            1.0 + self.gate_scale * (gate - 0.5) * 2.0
        )
        batch_dict['wing_gate_mean'] = gate.mean().detach()
        return batch_dict


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
