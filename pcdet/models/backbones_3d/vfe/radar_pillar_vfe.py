import torch

from .vfe_template import VFETemplate


class RadarPillarVFE(VFETemplate):
    """Aggregate voxel points while preserving radar return statistics."""

    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = int(num_point_features)
        self.stat_feature_indices = tuple(
            int(index) for index in self.model_cfg.get('STAT_FEATURE_INDICES', [])
        )
        self.include_point_count = bool(
            self.model_cfg.get('INCLUDE_POINT_COUNT', True)
        )
        self.max_points_per_voxel = float(
            self.model_cfg.get('MAX_POINTS_PER_VOXEL', 1)
        )
        if self.max_points_per_voxel <= 0:
            raise ValueError('MAX_POINTS_PER_VOXEL must be positive')
        for index in self.stat_feature_indices:
            if index < 0 or index >= self.num_point_features:
                raise ValueError(
                    'STAT_FEATURE_INDICES contains %d for %d input features'
                    % (index, self.num_point_features)
                )

        self.output_feature_dim = self.num_point_features
        self.output_feature_dim += 2 * len(self.stat_feature_indices)
        if self.include_point_count:
            self.output_feature_dim += 1

    def get_output_feature_dim(self):
        return self.output_feature_dim

    def forward(self, batch_dict, **kwargs):
        voxel_features = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        max_points = voxel_features.shape[1]

        point_indices = torch.arange(
            max_points, device=voxel_features.device
        ).view(1, -1)
        valid_mask = point_indices < voxel_num_points.view(-1, 1)
        valid_mask_float = valid_mask.unsqueeze(-1).type_as(voxel_features)
        normalizer = voxel_num_points.clamp_min(1).view(-1, 1).type_as(
            voxel_features
        )

        point_means = (voxel_features * valid_mask_float).sum(dim=1)
        point_means = point_means / normalizer
        output_features = [point_means]

        if self.stat_feature_indices:
            physics = voxel_features[:, :, self.stat_feature_indices]
            physics_mask = valid_mask.unsqueeze(-1)
            physics_means = point_means[:, self.stat_feature_indices].unsqueeze(1)
            centered = (physics - physics_means) * physics_mask.type_as(physics)
            physics_std = torch.sqrt(
                (centered.square().sum(dim=1) / normalizer).clamp_min(0.0)
            )
            physics_max = physics.masked_fill(
                ~physics_mask, float('-inf')
            ).amax(dim=1)
            physics_max = torch.where(
                torch.isfinite(physics_max),
                physics_max,
                torch.zeros_like(physics_max),
            )
            output_features.extend([physics_max, physics_std])

        if self.include_point_count:
            count_fraction = (
                voxel_num_points.view(-1, 1).type_as(voxel_features)
                / self.max_points_per_voxel
            )
            output_features.append(count_fraction)

        batch_dict['voxel_features'] = torch.cat(
            output_features, dim=1
        ).contiguous()
        return batch_dict
