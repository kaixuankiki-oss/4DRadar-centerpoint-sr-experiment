import torch
import torch.nn as nn


class DopplerTemporalConsistencyGate(nn.Module):
    """FRONT5 doppler/radial temporal consistency gate for pillar tokens.

    The module is inserted after RadialRCSConsistencyPillarVFE and before
    RadarPhysGatedPillarTransformer. It reconstructs approximate sweep bins
    from the timestamp feature, computes per-pillar temporal range/doppler
    consistency statistics, and applies an identity-initialized residual
    scale/bias gate to pillar tokens.
    """

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.enabled = bool(model_cfg.get('ENABLED', True))
        self.num_point_features = input_channels
        self.backbone_channels = None

        self.hidden_dim = int(model_cfg.get('HIDDEN_DIM', 64))
        self.dropout = float(model_cfg.get('DROPOUT', 0.05))
        self.scale_limit = float(model_cfg.get('SCALE_LIMIT', 0.20))
        self.bias_limit = float(model_cfg.get('BIAS_LIMIT', 0.10))
        self.velocity_clamp = float(model_cfg.get('VELOCITY_CLAMP', 40.0))
        self.rcs_clamp = float(model_cfg.get('RCS_CLAMP', 80.0))
        self.power_clamp = float(model_cfg.get('POWER_CLAMP', 80.0))
        self.range_norm = float(model_cfg.get('RANGE_NORM', 200.0))
        self.current_time_eps = float(model_cfg.get('CURRENT_TIME_EPS', 1e-3))
        self.sweep_time_bin = float(model_cfg.get('SWEEP_TIME_BIN', 0.1))
        self.num_sweep_bins = int(model_cfg.get('NUM_SWEEP_BINS', 5))
        self.log_telemetry = bool(model_cfg.get('LOG_TELEMETRY', True))
        self.last_telemetry = {}

        self.voxel_x = float(voxel_size[0])
        self.voxel_y = float(voxel_size[1])
        self.x_offset = self.voxel_x / 2.0 + float(point_cloud_range[0])
        self.y_offset = self.voxel_y / 2.0 + float(point_cloud_range[1])
        edges = list(model_cfg.get('BUCKET_EDGES', [0, 50, 100, 150, 200]))
        self.register_buffer('bucket_edges', torch.tensor(edges, dtype=torch.float32), persistent=False)

        stat_dim = 27
        self.mlp = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, input_channels * 2),
        )
        if bool(model_cfg.get('INIT_IDENTITY', True)):
            final = self.mlp[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @staticmethod
    def _valid_mask(voxels, voxel_num_points):
        arange = torch.arange(voxels.shape[1], device=voxels.device)[None, :]
        return (arange < voxel_num_points[:, None]).to(voxels.dtype).unsqueeze(-1)

    @staticmethod
    def _masked_mean(values, mask):
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (values * mask).sum(dim=1) / denom

    def _masked_std(self, values, mask):
        mean = self._masked_mean(values, mask).unsqueeze(1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        var = (((values - mean) * mask) ** 2).sum(dim=1) / denom
        return torch.sqrt(var.clamp_min(1e-6))

    def _current_history_delta(self, values, timestamp, valid):
        current_mask = (timestamp.abs() <= self.current_time_eps).to(values.dtype) * valid
        history_mask = (timestamp.abs() > self.current_time_eps).to(values.dtype) * valid
        cur_has = (current_mask.sum(dim=1, keepdim=True) > 0).to(values.dtype)
        hist_has = (history_mask.sum(dim=1, keepdim=True) > 0).to(values.dtype)
        current_mask = current_mask * cur_has + valid * (1.0 - cur_has)
        history_mask = history_mask * hist_has + valid * (1.0 - hist_has)
        return (self._masked_mean(values, current_mask) - self._masked_mean(values, history_mask)).abs()

    def _bin_means(self, values, sweep_ids, valid):
        one_hot = torch.nn.functional.one_hot(sweep_ids, num_classes=self.num_sweep_bins).to(values.dtype)
        weights = one_hot * valid.squeeze(-1).unsqueeze(-1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        means = torch.einsum('mpb,mpc->mbc', weights, values) / counts.unsqueeze(-1)
        present = weights.sum(dim=1) > 0
        return means, present, weights.sum(dim=1)

    def _temporal_stats(self, point_range, doppler, radial_from_vxy, rcs, power, timestamp, valid):
        sweep_ids = torch.round(timestamp.squeeze(-1) / self.sweep_time_bin).long()
        sweep_ids = sweep_ids.clamp_(0, self.num_sweep_bins - 1)
        packed = torch.cat([point_range, doppler, radial_from_vxy, rcs, power], dim=-1)
        bin_mean, present, raw_counts = self._bin_means(packed, sweep_ids, valid)

        bin_range = bin_mean[:, :, 0]
        bin_doppler = bin_mean[:, :, 1]
        bin_radial = bin_mean[:, :, 2]
        bin_rcs = bin_mean[:, :, 3]
        bin_power = bin_mean[:, :, 4]

        overall_range = self._masked_mean(point_range, valid).squeeze(-1)
        has_current = present[:, 0]
        current_range = torch.where(has_current, bin_range[:, 0], overall_range).unsqueeze(-1)
        range_delta = bin_range - current_range

        dt = torch.arange(self.num_sweep_bins, device=point_range.device, dtype=point_range.dtype) * self.sweep_time_bin
        doppler_disp = bin_doppler * dt.unsqueeze(0)
        radial_disp = bin_radial * dt.unsqueeze(0)
        hist_mask = present.clone()
        hist_mask[:, 0] = False
        hist_weight = hist_mask.to(point_range.dtype)
        hist_denom = hist_weight.sum(dim=1).clamp_min(1.0)

        # Sign can vary with radar convention, so compare displacement magnitudes.
        doppler_residual = (range_delta.abs() - doppler_disp.abs()).abs()
        radial_residual = (range_delta.abs() - radial_disp.abs()).abs()
        temporal_range_residual_mean = (doppler_residual * hist_weight).sum(dim=1, keepdim=True) / hist_denom.unsqueeze(-1)
        temporal_range_residual_max = (doppler_residual * hist_weight).max(dim=1, keepdim=True).values
        temporal_radial_residual_mean = (radial_residual * hist_weight).sum(dim=1, keepdim=True) / hist_denom.unsqueeze(-1)

        active_bins = present.sum(dim=1, keepdim=True).to(point_range.dtype)
        history_points = raw_counts[:, 1:].sum(dim=1, keepdim=True)
        total_points = raw_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        present_f = present.to(point_range.dtype)
        time_span = (present_f * dt.unsqueeze(0)).max(dim=1, keepdim=True).values
        large = point_range.new_tensor(1e6)
        range_max = bin_range.masked_fill(~present, -large).max(dim=1, keepdim=True).values
        range_min = bin_range.masked_fill(~present, large).min(dim=1, keepdim=True).values
        rcs_max = bin_rcs.masked_fill(~present, -large).max(dim=1, keepdim=True).values
        rcs_min = bin_rcs.masked_fill(~present, large).min(dim=1, keepdim=True).values
        power_max = bin_power.masked_fill(~present, -large).max(dim=1, keepdim=True).values
        power_min = bin_power.masked_fill(~present, large).min(dim=1, keepdim=True).values
        has_any_bin = active_bins > 0
        range_span = torch.where(has_any_bin, (range_max - range_min).abs(), torch.zeros_like(range_max))
        bin_doppler_mean = (bin_doppler * present_f).sum(dim=1, keepdim=True) / active_bins.clamp_min(1.0)
        bin_doppler_var = (((bin_doppler - bin_doppler_mean) * present_f) ** 2).sum(dim=1, keepdim=True) / active_bins.clamp_min(1.0)
        bin_doppler_std = torch.sqrt(bin_doppler_var.clamp_min(1e-6))
        rcs_sweep_span = torch.where(has_any_bin, (rcs_max - rcs_min).abs(), torch.zeros_like(rcs_max))
        power_sweep_span = torch.where(has_any_bin, (power_max - power_min).abs(), torch.zeros_like(power_max))

        return {
            'active_bin_frac': active_bins / float(self.num_sweep_bins),
            'history_point_frac': history_points / total_points,
            'time_span_norm': time_span / max(self.sweep_time_bin * max(self.num_sweep_bins - 1, 1), 1e-3),
            'range_span_norm': range_span / self.range_norm,
            'temporal_range_residual_mean': temporal_range_residual_mean / self.range_norm,
            'temporal_range_residual_max': temporal_range_residual_max / self.range_norm,
            'temporal_radial_residual_mean': temporal_radial_residual_mean / self.range_norm,
            'bin_doppler_std': bin_doppler_std / self.velocity_clamp,
            'rcs_sweep_span': rcs_sweep_span / self.rcs_clamp,
            'power_sweep_span': power_sweep_span / self.power_clamp,
        }

    def _bucket_ids(self, coords, dtype):
        x = coords[:, 3].to(dtype) * self.voxel_x + self.x_offset
        y = coords[:, 2].to(dtype) * self.voxel_y + self.y_offset
        ranges = torch.sqrt(torch.clamp(x * x + y * y, min=0.0))
        inner_edges = self.bucket_edges[1:-1].to(device=ranges.device, dtype=ranges.dtype)
        return torch.bucketize(ranges, inner_edges, right=False).clamp_(0, len(self.bucket_edges) - 2).long()

    def _stats(self, voxels, voxel_num_points):
        if voxels.shape[-1] < 9:
            raise ValueError('DopplerTemporalConsistencyGate expects 9D radar features')

        valid = self._valid_mask(voxels, voxel_num_points)
        xy = voxels[:, :, 0:2]
        point_range = torch.norm(xy, dim=-1, keepdim=True).clamp_min(1e-3)
        radial_unit = xy / point_range

        timestamp = voxels[:, :, 3:4]
        rcs = voxels[:, :, 4:5].clamp(-self.rcs_clamp, self.rcs_clamp)
        power = voxels[:, :, 5:6].clamp(-self.power_clamp, self.power_clamp)
        doppler = voxels[:, :, 6:7].clamp(-self.velocity_clamp, self.velocity_clamp)
        vxy = voxels[:, :, 7:9].clamp(-self.velocity_clamp, self.velocity_clamp)

        radial_from_vxy = (vxy * radial_unit).sum(dim=-1, keepdim=True)
        radial_delta = radial_from_vxy - doppler
        tangential = vxy - radial_from_vxy * radial_unit
        tangential_mag = torch.norm(tangential, dim=-1, keepdim=True)
        speed_xy = torch.norm(vxy, dim=-1, keepdim=True)
        rcs_power_delta = self._current_history_delta(torch.cat([rcs, power], dim=-1), timestamp, valid)
        temporal = self._temporal_stats(point_range, doppler, radial_from_vxy, rcs, power, timestamp, valid)

        stats = torch.cat([
            self._masked_mean(doppler, valid) / self.velocity_clamp,
            self._masked_mean(doppler.abs(), valid) / self.velocity_clamp,
            self._masked_std(doppler, valid) / self.velocity_clamp,
            self._masked_mean(radial_from_vxy, valid) / self.velocity_clamp,
            self._masked_mean(radial_delta, valid) / self.velocity_clamp,
            self._masked_mean(radial_delta.abs(), valid) / self.velocity_clamp,
            self._masked_std(radial_delta, valid) / self.velocity_clamp,
            self._masked_mean(tangential_mag, valid) / self.velocity_clamp,
            self._masked_mean(speed_xy, valid) / self.velocity_clamp,
            self._masked_mean(vxy, valid) / self.velocity_clamp,
            self._masked_mean(rcs, valid) / self.rcs_clamp,
            self._masked_std(rcs, valid) / self.rcs_clamp,
            self._masked_mean(power, valid) / self.power_clamp,
            self._masked_std(power, valid) / self.power_clamp,
            rcs_power_delta[:, 0:1] / self.rcs_clamp,
            rcs_power_delta[:, 1:2] / self.power_clamp,
            voxel_num_points.to(voxels.dtype).unsqueeze(-1) / float(voxels.shape[1]),
            temporal['active_bin_frac'],
            temporal['history_point_frac'],
            temporal['time_span_norm'],
            temporal['range_span_norm'],
            temporal['temporal_range_residual_mean'],
            temporal['temporal_range_residual_max'],
            temporal['temporal_radial_residual_mean'],
            temporal['bin_doppler_std'],
            temporal['rcs_sweep_span'],
        ], dim=-1)
        return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0), temporal

    def forward(self, batch_dict):
        if not self.enabled:
            return batch_dict
        pillar_features = batch_dict['pillar_features']
        if pillar_features.shape[0] == 0:
            return batch_dict

        stats, temporal = self._stats(batch_dict['voxels'], batch_dict['voxel_num_points'])
        scale_bias = self.mlp(stats)
        scale_raw, bias_raw = torch.chunk(scale_bias, 2, dim=-1)
        scale = torch.tanh(scale_raw) * self.scale_limit
        bias = torch.tanh(bias_raw) * self.bias_limit
        out = pillar_features * (1.0 + scale) + bias
        batch_dict['pillar_features'] = out

        if self.log_telemetry:
            with torch.no_grad():
                delta_norm = (out - pillar_features).norm(dim=-1).mean()
                parent_norm = pillar_features.norm(dim=-1).mean().clamp_min(1e-6)
                bucket_ids = self._bucket_ids(batch_dict['voxel_coords'], pillar_features.dtype)
                bucket_scale = []
                bucket_delta = []
                for bid in range(len(self.bucket_edges) - 1):
                    mask = bucket_ids == bid
                    bucket_scale.append(scale[mask].abs().mean().detach() if mask.any() else scale.new_tensor(0.0))
                    bucket_delta.append((out[mask] - pillar_features[mask]).norm(dim=-1).mean().detach() if mask.any() else scale.new_tensor(0.0))
                telemetry = {
                    'doppler_abs_mean': stats[:, 1].mean().detach(),
                    'radial_delta_abs_mean': stats[:, 5].mean().detach(),
                    'active_bin_frac': stats[:, 17].mean().detach(),
                    'history_point_frac': stats[:, 18].mean().detach(),
                    'time_span_norm': stats[:, 19].mean().detach(),
                    'temporal_range_residual_mean': stats[:, 21].mean().detach(),
                    'temporal_radial_residual_mean': stats[:, 23].mean().detach(),
                    'bin_doppler_std': stats[:, 24].mean().detach(),
                    'scale_abs_mean': scale.abs().mean().detach(),
                    'bias_abs_mean': bias.abs().mean().detach(),
                    'feature_delta_norm': delta_norm.detach(),
                    'feature_delta_ratio': (delta_norm / parent_norm).detach(),
                    'finite': torch.isfinite(out).all().detach(),
                    'scale_abs_mean_by_bucket': torch.stack(bucket_scale),
                    'feature_delta_norm_by_bucket': torch.stack(bucket_delta),
                }
                self.last_telemetry = telemetry
                batch_dict['doppler_temporal_consistency_telemetry'] = telemetry
        return batch_dict
