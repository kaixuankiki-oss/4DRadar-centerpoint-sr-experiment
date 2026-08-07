import torch
import torch.nn as nn


class RangeBucketModulation(nn.Module):
    """Range-aware pillar token modulation for 4D Radar CenterPoint.

    This module operates on PillarVFE tokens after the physical gated transformer
    and before PointPillarScatter. It keeps the detector head and BEV backbone
    unchanged while adding distance-bucket conditioning to pillar features.
    """

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.enabled = bool(model_cfg.get('ENABLED', True))
        self.mode = model_cfg.get('MODE', 'feature_modulation')
        self.apply_to = model_cfg.get('APPLY_TO', 'pillar_tokens')
        self.use_distance = model_cfg.get('USE_DISTANCE', 'euclidean_xy')
        self.alpha = float(model_cfg.get('ALPHA', 0.25))
        self.beta = float(model_cfg.get('BETA', 0.10))
        self.num_point_features = input_channels
        self.backbone_channels = None

        self.voxel_x = float(voxel_size[0])
        self.voxel_y = float(voxel_size[1])
        self.x_offset = self.voxel_x / 2.0 + float(point_cloud_range[0])
        self.y_offset = self.voxel_y / 2.0 + float(point_cloud_range[1])

        if self.apply_to != 'pillar_tokens':
            raise ValueError(f'RangeBucketModulation only supports pillar_tokens, got {self.apply_to}')
        if self.use_distance != 'euclidean_xy':
            raise ValueError(f'RangeBucketModulation only supports euclidean_xy, got {self.use_distance}')

        edges = list(model_cfg.get('BUCKET_EDGES', [0, 50, 100, 150, 200]))
        if len(edges) < 2:
            raise ValueError('BUCKET_EDGES must contain at least two values')
        self.register_buffer('bucket_edges', torch.tensor(edges, dtype=torch.float32), persistent=False)

        embed_dim = int(model_cfg.get('EMBED_DIM', 16))
        hidden_dim = int(model_cfg.get('HIDDEN_DIM', input_channels))
        dropout = float(model_cfg.get('DROPOUT', 0.05))
        num_buckets = len(edges) - 1

        self.range_embedding = nn.Embedding(num_buckets, embed_dim)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(input_channels + embed_dim),
            nn.Linear(input_channels + embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_channels),
            nn.Sigmoid(),
        )
        self.range_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, input_channels),
        )

    def _pillar_centers(self, coords, dtype):
        # coords layout is batch, z, y, x. Voxel size is already encoded by grid index in BEV.
        x_idx = coords[:, 3].to(dtype)
        y_idx = coords[:, 2].to(dtype)
        # Center coordinates for baseline ROI/voxelization.
        # point_cloud_range and voxel_size are stored as Python/numpy inputs at construction.
        x_center = x_idx * self.voxel_x + self.x_offset
        y_center = y_idx * self.voxel_y + self.y_offset
        return x_center, y_center

    def _bucket_ids(self, ranges):
        inner_edges = self.bucket_edges[1:-1].to(device=ranges.device, dtype=ranges.dtype)
        bucket_ids = torch.bucketize(ranges, inner_edges, right=False)
        return bucket_ids.clamp_(0, self.range_embedding.num_embeddings - 1).long()

    def forward(self, batch_dict):
        if not self.enabled:
            return batch_dict

        pillar_features = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']
        if pillar_features.shape[0] == 0:
            batch_dict['range_bucket_ids'] = coords.new_zeros((0,), dtype=torch.long)
            batch_dict['range_bucket_hist'] = coords.new_zeros((self.range_embedding.num_embeddings,), dtype=torch.long)
            return batch_dict

        x_center, y_center = self._pillar_centers(coords, pillar_features.dtype)
        ranges = torch.sqrt(torch.clamp(x_center * x_center + y_center * y_center, min=0.0))
        bucket_ids = self._bucket_ids(ranges)
        range_emb = self.range_embedding(bucket_ids)

        if self.mode == 'residual':
            out = pillar_features + self.beta * self.range_projection(range_emb)
        elif self.mode == 'feature_modulation':
            gate_input = torch.cat([pillar_features, range_emb], dim=-1)
            gate = self.gate_mlp(gate_input)
            out = pillar_features * (1.0 + self.alpha * gate) + self.beta * self.range_projection(range_emb)
        else:
            raise ValueError(f'Unsupported RangeBucketModulation MODE: {self.mode}')

        batch_dict['pillar_features'] = out
        batch_dict['range_bucket_ids'] = bucket_ids
        batch_dict['range_bucket_hist'] = torch.bincount(bucket_ids, minlength=self.range_embedding.num_embeddings)
        return batch_dict
