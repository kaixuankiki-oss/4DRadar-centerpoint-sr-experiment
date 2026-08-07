import torch
import torch.nn as nn


class RadarPhysicalGatingBlock(nn.Module):
    def __init__(self, channels, model_cfg):
        super().__init__()
        self.amp_indices = list(model_cfg.get('AMP_FEATURE_INDICES', [3, 4]))
        self.motion_indices = list(model_cfg.get('MOTION_FEATURE_INDICES', [5, 6, 7]))
        self.gate_scale = float(model_cfg.get('GATE_SCALE', 0.5))
        hidden_dim = int(model_cfg.get('GATE_HIDDEN_DIM', channels))

        self.amp_gate = self._make_gate(len(self.amp_indices) * 2, hidden_dim, channels)
        self.motion_gate = self._make_gate(len(self.motion_indices) * 2, hidden_dim, channels)
        self.fusion = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    @staticmethod
    def _make_gate(in_channels, hidden_dim, out_channels):
        return nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
            nn.Sigmoid(),
        )

    @staticmethod
    def _masked_stats(voxels, voxel_num_points, indices):
        if len(indices) == 0:
            return voxels.new_zeros((voxels.shape[0], 0))

        max_index = max(indices)
        if voxels.shape[-1] <= max_index:
            raise ValueError(
                f'Radar physical feature index {max_index} exceeds voxel feature dim {voxels.shape[-1]}'
            )

        values = voxels[..., indices]
        valid_mask = (
            torch.arange(voxels.shape[1], device=voxels.device)[None, :] < voxel_num_points[:, None]
        ).to(values.dtype)
        denom = voxel_num_points.clamp_min(1).to(values.dtype).unsqueeze(-1)
        mean = (values * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
        abs_mean = (values.abs() * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
        return torch.cat([mean, abs_mean], dim=-1)

    def forward(self, pillar_features, voxels, voxel_num_points):
        amp_stats = self._masked_stats(voxels, voxel_num_points, self.amp_indices)
        motion_stats = self._masked_stats(voxels, voxel_num_points, self.motion_indices)
        amp_gate = self.amp_gate(amp_stats)
        motion_gate = self.motion_gate(motion_stats)
        gate = 1.0 + self.gate_scale * (amp_gate + motion_gate - 1.0)
        gated_features = pillar_features * gate
        return gated_features + self.fusion(gated_features)


class SerializedPillarTransformer(nn.Module):
    def __init__(self, channels, model_cfg):
        super().__init__()
        transformer_cfg = model_cfg.get('TRANSFORMER', {})
        self.enabled = bool(transformer_cfg.get('ENABLED', True))
        self.patch_size = int(transformer_cfg.get('PATCH_SIZE', 256))
        num_layers = int(transformer_cfg.get('NUM_LAYERS', 1))
        num_heads = int(transformer_cfg.get('NUM_HEADS', 4))
        ffn_dim = int(transformer_cfg.get('FFN_DIM', channels * 2))
        dropout = float(transformer_cfg.get('DROPOUT', 0.05))

        self.layers = nn.ModuleList()
        if self.enabled and num_layers > 0:
            self.layers.extend([
                nn.TransformerEncoderLayer(
                    d_model=channels,
                    nhead=num_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ])
            self.out_norm = nn.LayerNorm(channels)
        else:
            self.out_norm = nn.Identity()

    @staticmethod
    def _sort_indices(coords):
        # coords follow OpenPCDet layout: batch, z, y, x. Keep nearby BEV pillars together.
        if coords.numel() == 0:
            return coords.new_zeros((0,), dtype=torch.long)
        batch = coords[:, 0].to(torch.long)
        y = coords[:, 2].to(torch.long)
        x = coords[:, 3].to(torch.long)
        linear = batch * 1_000_000_000 + y * 1_000_000 + x
        return torch.argsort(linear)

    def forward(self, pillar_features, coords):
        if not self.layers or pillar_features.shape[0] == 0:
            return pillar_features

        order = self._sort_indices(coords)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.shape[0], device=order.device)
        sorted_features = pillar_features[order]

        outputs = []
        for start in range(0, sorted_features.shape[0], self.patch_size):
            chunk = sorted_features[start:start + self.patch_size].unsqueeze(0)
            residual = chunk
            for layer in self.layers:
                chunk = layer(chunk)
            outputs.append((residual + self.out_norm(chunk)).squeeze(0))

        return torch.cat(outputs, dim=0)[inverse]


class RadarPhysGatedPillarTransformer(nn.Module):
    """Radar-specific pillar token enhancer inserted after PillarVFE."""

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_point_features = input_channels
        self.backbone_channels = None
        self.physical_gate = RadarPhysicalGatingBlock(input_channels, model_cfg)
        self.transformer = SerializedPillarTransformer(input_channels, model_cfg)

    def forward(self, batch_dict):
        pillar_features = batch_dict['pillar_features']
        voxels = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        coords = batch_dict['voxel_coords']
        pillar_features = self.physical_gate(pillar_features, voxels, voxel_num_points)
        pillar_features = self.transformer(pillar_features, coords)
        batch_dict['pillar_features'] = pillar_features
        return batch_dict
