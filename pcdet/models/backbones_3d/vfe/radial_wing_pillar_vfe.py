import torch
import torch.nn as nn

from .pillar_vfe import PillarVFE


class RangeTimeGatePillarVFE(PillarVFE):
    """PillarVFE with a learned range/time/RCS/Doppler point gate.

    This gate treats Vx/Vy as radar radial-velocity decompositions only. It does
    not use them for object-motion compensation.
    """

    DEFAULT_FEATURE_NAMES = ['x', 'y', 'z', 'RCS', 'power', 'doppler', 'Vx', 'Vy', 'timestamp']
    AMPLITUDE_CANDIDATES = ['RCS', 'power']

    def __init__(
        self,
        model_cfg,
        num_point_features,
        voxel_size,
        point_cloud_range,
        point_feature_names=None,
        **kwargs
    ):
        super().__init__(
            model_cfg=model_cfg,
            num_point_features=num_point_features,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            **kwargs,
        )
        self._init_feature_indices(num_point_features, point_feature_names)
        self.gate_scale = float(self.model_cfg.get('GATE_SCALE', 0.5))
        self.range_norm = float(self.model_cfg.get('RANGE_NORM', 200.0))
        gate_hidden = int(self.model_cfg.get('GATE_HIDDEN_DIM', 16))
        gate_in = self.base_gate_input_dim
        self.point_gate = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

    def _init_feature_indices(self, num_point_features, point_feature_names):
        if point_feature_names is None:
            point_feature_names = self.model_cfg.get('FEATURE_NAMES', None)
        if point_feature_names is None:
            point_feature_names = self.DEFAULT_FEATURE_NAMES[:num_point_features]

        self.point_feature_names = list(point_feature_names)
        if len(self.point_feature_names) != num_point_features:
            raise ValueError(
                'point_feature_names length %d does not match num_point_features %d'
                % (len(self.point_feature_names), num_point_features)
            )

        self.feature_to_idx = {}
        for idx, name in enumerate(self.point_feature_names):
            if name in self.feature_to_idx:
                raise ValueError('duplicate point feature name: %s' % name)
            self.feature_to_idx[name] = idx

        self.time_feature = self.model_cfg.get('TIME_FEATURE', 'timestamp')
        self.doppler_feature = self.model_cfg.get('DOPPLER_FEATURE', 'doppler')
        self.x_feature = self.model_cfg.get('X_FEATURE', 'x')
        self.y_feature = self.model_cfg.get('Y_FEATURE', 'y')
        self.z_feature = self.model_cfg.get('Z_FEATURE', 'z')
        self.vx_feature = self.model_cfg.get('VX_FEATURE', 'Vx')
        self.vy_feature = self.model_cfg.get('VY_FEATURE', 'Vy')

        amplitude_features = [
            name for name in self.AMPLITUDE_CANDIDATES if name in self.feature_to_idx
        ]

        if not amplitude_features:
            raise KeyError(
                'no amplitude feature is available; add RCS or power to used_feature_list'
            )
        self._require_features(
            [
                self.x_feature,
                self.y_feature,
                self.z_feature,
                self.time_feature,
                self.doppler_feature,
            ] + amplitude_features
        )
        self.amplitude_features = amplitude_features
        self.base_gate_input_dim = 2 + len(self.amplitude_features) + 1
        self.temporal_input_dim = 3 * len(self.amplitude_features)

    def _require_features(self, names):
        missing = [name for name in names if name not in self.feature_to_idx]
        if missing:
            raise KeyError(
                'missing point feature(s) %s in %s'
                % (missing, self.point_feature_names)
            )

    def _feature(self, voxel_features, name):
        idx = self.feature_to_idx[name]
        return voxel_features[:, :, idx:idx + 1]

    def _features(self, voxel_features, names):
        return torch.cat([self._feature(voxel_features, name) for name in names], dim=-1)

    def _gate_inputs(self, voxel_features, voxel_num_points=None):
        xy = self._features(voxel_features, [self.x_feature, self.y_feature])
        xy_range = torch.norm(xy, dim=-1, keepdim=True) / self.range_norm
        timestamp = self._feature(voxel_features, self.time_feature)
        doppler = self._feature(voxel_features, self.doppler_feature)
        gate_parts = [xy_range, timestamp]
        if 'RCS' in self.amplitude_features:
            gate_parts.append(self._feature(voxel_features, 'RCS'))
            used_amplitude_features = {'RCS'}
        else:
            gate_parts.append(self._feature(voxel_features, self.amplitude_features[0]))
            used_amplitude_features = {self.amplitude_features[0]}
        gate_parts.append(doppler)
        gate_parts.extend(
            self._feature(voxel_features, name)
            for name in self.amplitude_features
            if name not in used_amplitude_features
        )
        return torch.cat(gate_parts, dim=-1)

    def _apply_point_gate(self, voxel_features, voxel_num_points):
        gate = self.point_gate(self._gate_inputs(voxel_features, voxel_num_points))
        point_weight = 1.0 + self.gate_scale * (gate - 0.5) * 2.0
        padded = self.get_paddings_indicator(
            voxel_num_points, voxel_features.shape[1], axis=0
        ).unsqueeze(-1).type_as(voxel_features)
        feature_weight = torch.ones_like(voxel_features)
        # Preserve geometry and sweep time; gate only radar evidence features.
        protected_features = {
            self.x_feature, self.y_feature, self.z_feature, self.time_feature
        }
        for idx, name in enumerate(self.point_feature_names):
            if name not in protected_features:
                feature_weight[:, :, idx:idx + 1] = point_weight
        return voxel_features * feature_weight * padded

    def _raw_feature_weight(self, voxel_features, voxel_num_points):
        gate = self.point_gate(self._gate_inputs(voxel_features, voxel_num_points))
        point_weight = 1.0 + self.gate_scale * (gate - 0.5) * 2.0
        feature_weight = torch.ones_like(voxel_features)
        # Preserve geometry and sweep time; gate only radar evidence features.
        protected_features = {
            self.x_feature, self.y_feature, self.z_feature, self.time_feature
        }
        for idx, name in enumerate(self.point_feature_names):
            if name not in protected_features:
                feature_weight[:, :, idx:idx + 1] = point_weight
        return feature_weight

    def _gate_inputs_export(self, voxel_features, valid_mask):
        return RangeTimeGatePillarVFE._gate_inputs(self, voxel_features, None)

    def _gated_raw_features_export(self, voxel_features, valid_mask):
        gate = self.point_gate(self._gate_inputs_export(voxel_features, valid_mask))
        point_weight = 1.0 + self.gate_scale * (gate - 0.5) * 2.0
        protected_features = {
            self.x_feature, self.y_feature, self.z_feature, self.time_feature
        }
        gated_parts = []
        for idx, name in enumerate(self.point_feature_names):
            feature = voxel_features[:, :, idx:idx + 1]
            if name in protected_features:
                gated_parts.append(feature)
            else:
                gated_parts.append(feature * point_weight)
        return torch.cat(gated_parts, dim=-1) * valid_mask

    def forward_export_onnx(self, batch_dict):
        pfn_features = batch_dict['voxels_cart']
        raw_dim = len(self.point_feature_names)
        voxel_features = pfn_features[:, :, :raw_dim]
        valid_mask = (voxel_features.abs().sum(dim=-1, keepdim=True) > 0).type_as(pfn_features)
        gated_raw_features = self._gated_raw_features_export(voxel_features, valid_mask)

        if self.use_absolute_xyz:
            features = torch.cat([gated_raw_features, pfn_features[:, :, raw_dim:]], dim=-1)
        else:
            features = torch.cat([gated_raw_features[:, :, 3:], pfn_features[:, :, raw_dim - 3:]], dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict

    def forward(self, batch_dict, **kwargs):
        if getattr(self, 'export_onnx', False) and 'voxels_cart' in batch_dict:
            return self.forward_export_onnx(batch_dict)

        voxel_features, voxel_num_points, coords = (
            batch_dict['voxels'],
            batch_dict['voxel_num_points'],
            batch_dict['voxel_coords'],
        )
        raw_feature_weight = self._raw_feature_weight(voxel_features, voxel_num_points)
        padded = self.get_paddings_indicator(
            voxel_num_points, voxel_features.shape[1], axis=0
        ).unsqueeze(-1).type_as(voxel_features)
        voxel_features = voxel_features * padded

        points_mean = (
            voxel_features[:, :, :3].sum(dim=1, keepdim=True)
            / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        )
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = (
            voxel_features[:, :, 0]
            - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        )
        f_center[:, :, 1] = (
            voxel_features[:, :, 1]
            - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        )
        f_center[:, :, 2] = (
            voxel_features[:, :, 2]
            - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        )

        gated_voxel_features = voxel_features * raw_feature_weight
        if self.use_absolute_xyz:
            features = [gated_voxel_features, f_cluster, f_center]
        else:
            features = [gated_voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict


class RadialConsistencyPillarVFE(RangeTimeGatePillarVFE):
    """Range/time gate with radial-consistency inputs.

    Vx/Vy are interpreted only as decomposed radial velocity. The tangential
    residual is used as an uncertainty cue, not as true lateral object velocity.
    """

    def __init__(
        self,
        model_cfg,
        num_point_features,
        voxel_size,
        point_cloud_range,
        point_feature_names=None,
        **kwargs
    ):
        super().__init__(
            model_cfg=model_cfg,
            num_point_features=num_point_features,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            point_feature_names=point_feature_names,
            **kwargs,
        )
        self._require_features([self.vx_feature, self.vy_feature])
        gate_hidden = int(self.model_cfg.get('GATE_HIDDEN_DIM', 16))
        self.point_gate = nn.Sequential(
            nn.Linear(self.base_gate_input_dim + 3, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

    def _radial_inputs(self, voxel_features):
        xy = self._features(voxel_features, [self.x_feature, self.y_feature])
        xy_norm = torch.norm(xy, dim=-1, keepdim=True).clamp_min(1e-3)
        radial_unit = xy / xy_norm
        vxy = self._features(voxel_features, [self.vx_feature, self.vy_feature])
        radial_from_vxy = (vxy * radial_unit).sum(dim=-1, keepdim=True)
        tangential = vxy - radial_from_vxy * radial_unit
        tangential_mag = torch.norm(tangential, dim=-1, keepdim=True)
        doppler = self._feature(voxel_features, self.doppler_feature)
        radial_delta = radial_from_vxy - doppler
        return radial_from_vxy, radial_delta, tangential_mag

    def _gate_inputs(self, voxel_features, voxel_num_points=None):
        base_inputs = super()._gate_inputs(voxel_features, voxel_num_points)
        return torch.cat([base_inputs, *self._radial_inputs(voxel_features)], dim=-1)

    def _gate_inputs_export(self, voxel_features, valid_mask):
        base_inputs = RangeTimeGatePillarVFE._gate_inputs_export(self, voxel_features, valid_mask)
        return torch.cat([base_inputs, *self._radial_inputs(voxel_features)], dim=-1)


class _TemporalConsistencyMixin:
    def _valid_mask(self, voxel_features, voxel_num_points):
        if voxel_num_points is None:
            return torch.ones(
                voxel_features.shape[:2] + (1,),
                device=voxel_features.device,
                dtype=voxel_features.dtype,
            )
        return self.get_paddings_indicator(
            voxel_num_points, voxel_features.shape[1], axis=0
        ).unsqueeze(-1).type_as(voxel_features)

    @staticmethod
    def _masked_mean(values, mask):
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (values * mask).sum(dim=1, keepdim=True) / denom

    def _masked_std(self, values, mask):
        mean = self._masked_mean(values, mask)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        var = (((values - mean) * mask) ** 2).sum(dim=1, keepdim=True) / denom
        return torch.sqrt(var.clamp_min(1e-6))

    def _amplitude_temporal_inputs(self, voxel_features, voxel_num_points):
        valid = self._valid_mask(voxel_features, voxel_num_points)
        timestamp = self._feature(voxel_features, self.time_feature)
        current_eps = float(self.model_cfg.get('CURRENT_TIME_EPS', 1e-3))
        current_mask = (timestamp.abs() <= current_eps).type_as(voxel_features) * valid
        history_mask = (timestamp.abs() > current_eps).type_as(voxel_features) * valid
        # Fall back to all valid points if a pillar has no current/history points.
        current_has_points = (current_mask.sum(dim=1, keepdim=True) > 0).type_as(voxel_features)
        history_has_points = (history_mask.sum(dim=1, keepdim=True) > 0).type_as(voxel_features)
        current_mask = current_mask * current_has_points + valid * (1.0 - current_has_points)
        history_mask = history_mask * history_has_points + valid * (1.0 - history_has_points)

        amplitude_values = self._features(voxel_features, self.amplitude_features)
        current_mean = self._masked_mean(amplitude_values, current_mask)
        history_mean = self._masked_mean(amplitude_values, history_mask)
        history_std = self._masked_std(amplitude_values, history_mask)
        delta_current = (amplitude_values - current_mean).abs()
        delta_history = (amplitude_values - history_mean).abs()
        return torch.cat(
            [delta_current, delta_history, history_std.expand_as(amplitude_values)],
            dim=-1,
        )

    def _amplitude_temporal_inputs_export(self, voxel_features, valid):
        timestamp = self._feature(voxel_features, self.time_feature)
        current_eps = float(self.model_cfg.get('CURRENT_TIME_EPS', 1e-3))
        current_mask = (timestamp.abs() <= current_eps).type_as(voxel_features) * valid
        history_mask = (timestamp.abs() > current_eps).type_as(voxel_features) * valid

        current_has_points = (current_mask.sum(dim=1, keepdim=True) > 0).type_as(voxel_features)
        history_has_points = (history_mask.sum(dim=1, keepdim=True) > 0).type_as(voxel_features)
        current_mask = current_mask * current_has_points + valid * (1.0 - current_has_points)
        history_mask = history_mask * history_has_points + valid * (1.0 - history_has_points)

        amplitude_values = self._features(voxel_features, self.amplitude_features)
        current_mean = self._masked_mean(amplitude_values, current_mask)
        history_mean = self._masked_mean(amplitude_values, history_mask)
        history_std = self._masked_std(amplitude_values, history_mask)
        delta_current = (amplitude_values - current_mean).abs()
        delta_history = (amplitude_values - history_mean).abs()
        return torch.cat(
            [delta_current, delta_history, history_std.expand_as(amplitude_values)],
            dim=-1,
        )


class RCSTemporalConsistencyPillarVFE(_TemporalConsistencyMixin, RangeTimeGatePillarVFE):
    """RCS/power temporal gate without radial residual inputs."""

    def __init__(
        self,
        model_cfg,
        num_point_features,
        voxel_size,
        point_cloud_range,
        point_feature_names=None,
        **kwargs
    ):
        super().__init__(
            model_cfg=model_cfg,
            num_point_features=num_point_features,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            point_feature_names=point_feature_names,
            **kwargs,
        )
        gate_hidden = int(self.model_cfg.get('GATE_HIDDEN_DIM', 16))
        self.point_gate = nn.Sequential(
            nn.Linear(self.base_gate_input_dim + self.temporal_input_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

    def _gate_inputs(self, voxel_features, voxel_num_points=None):
        base_inputs = super()._gate_inputs(voxel_features, voxel_num_points)
        amplitude_temporal_inputs = self._amplitude_temporal_inputs(
            voxel_features, voxel_num_points
        )
        return torch.cat([base_inputs, amplitude_temporal_inputs], dim=-1)


class RadialRCSConsistencyPillarVFE(PillarVFE):
    """PillarVFE with radial and RCS/power temporal point gating.

    The gate is intentionally applied after cluster/center feature construction
    so the VFE can be split cleanly for deployment while preserving the training
    math of the original pre-VFE point gate.
    """

    FEATURE_NAMES = ['x', 'y', 'z', 'timestamp', 'RCS', 'power', 'doppler', 'Vx', 'Vy']
    X_IDX = 0
    Y_IDX = 1
    Z_IDX = 2
    TIME_IDX = 3
    RCS_IDX = 4
    POWER_IDX = 5
    DOPPLER_IDX = 6
    VX_IDX = 7
    VY_IDX = 8
    GATE_START_IDX = 4

    def __init__(
        self,
        model_cfg,
        num_point_features,
        voxel_size,
        point_cloud_range,
        point_feature_names=None,
        **kwargs
    ):
        super().__init__(
            model_cfg=model_cfg,
            num_point_features=num_point_features,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            **kwargs,
        )
        self._check_feature_layout(num_point_features, point_feature_names)
        self.gate_scale = float(self.model_cfg.get('GATE_SCALE', 0.5))
        self.range_norm = float(self.model_cfg.get('RANGE_NORM', 200.0))
        gate_hidden = int(self.model_cfg.get('GATE_HIDDEN_DIM', 32))
        gate_in = 14
        self.point_gate = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        gate_mask = torch.zeros(1, 1, 15)
        gate_mask[..., 4:9] = 1.0
        self.register_buffer("gate_mask", gate_mask)

    def _check_feature_layout(self, num_point_features, point_feature_names):
        if point_feature_names is None:
            point_feature_names = self.model_cfg.get('FEATURE_NAMES', None)
        if point_feature_names is None:
            point_feature_names = self.FEATURE_NAMES

        self.point_feature_names = list(point_feature_names)
        if num_point_features != len(self.FEATURE_NAMES) or self.point_feature_names != self.FEATURE_NAMES:
            raise ValueError(
                'RadialRCSConsistencyPillarVFE expects used_feature_list %s, got %s'
                % (self.FEATURE_NAMES, self.point_feature_names)
            )

    def _x(self, voxel_features):
        return voxel_features[:, :, self.X_IDX:self.X_IDX + 1]

    def _y(self, voxel_features):
        return voxel_features[:, :, self.Y_IDX:self.Y_IDX + 1]

    def _z(self, voxel_features):
        return voxel_features[:, :, self.Z_IDX:self.Z_IDX + 1]

    def _rcs(self, voxel_features):
        return voxel_features[:, :, self.RCS_IDX:self.RCS_IDX + 1]

    def _power(self, voxel_features):
        return voxel_features[:, :, self.POWER_IDX:self.POWER_IDX + 1]

    def _doppler(self, voxel_features):
        return voxel_features[:, :, self.DOPPLER_IDX:self.DOPPLER_IDX + 1]

    def _vx(self, voxel_features):
        return voxel_features[:, :, self.VX_IDX:self.VX_IDX + 1]

    def _vy(self, voxel_features):
        return voxel_features[:, :, self.VY_IDX:self.VY_IDX + 1]

    def _timestamp(self, voxel_features):
        return voxel_features[:, :, self.TIME_IDX:self.TIME_IDX + 1]

    def _xy(self, voxel_features):
        return voxel_features[:, :, self.X_IDX:self.Y_IDX + 1]

    def _vxy(self, voxel_features):
        return voxel_features[:, :, self.VX_IDX:self.VY_IDX + 1]

    def _amplitude_values(self, voxel_features):
        return voxel_features[:, :, self.RCS_IDX:self.POWER_IDX + 1]

    def _base_gate_inputs(self, voxel_features):
        xy = self._xy(voxel_features)
        xy_range = torch.norm(xy, dim=-1, keepdim=True) / self.range_norm
        return torch.cat(
            [
                xy_range,
                self._timestamp(voxel_features),
                self._rcs(voxel_features),
                self._doppler(voxel_features),
                self._power(voxel_features),
            ],
            dim=-1,
        )

    def _radial_inputs(self, voxel_features):
        xy = self._xy(voxel_features)
        xy_norm = torch.norm(xy, dim=-1, keepdim=True).clamp_min(1e-3)
        radial_unit = xy / xy_norm
        vxy = self._vxy(voxel_features)
        radial_from_vxy = (vxy * radial_unit).sum(dim=-1, keepdim=True)
        tangential = vxy - radial_from_vxy * radial_unit
        tangential_mag = torch.norm(tangential, dim=-1, keepdim=True)
        radial_delta = radial_from_vxy - self._doppler(voxel_features)
        return radial_from_vxy, radial_delta, tangential_mag

    def _valid_mask(self, voxel_features, voxel_num_points):
        if voxel_num_points is None:
            return torch.ones(
                voxel_features.shape[:2] + (1,),
                device=voxel_features.device,
                dtype=voxel_features.dtype,
            )
        return self.get_paddings_indicator(
            voxel_num_points, voxel_features.shape[1], axis=0
        ).unsqueeze(-1).type_as(voxel_features)

    @staticmethod
    def _masked_mean(values, mask):
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (values * mask).sum(dim=1, keepdim=True) / denom

    def _masked_std(self, values, mask):
        mean = self._masked_mean(values, mask)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        var = (((values - mean) * mask) ** 2).sum(dim=1, keepdim=True) / denom
        return torch.sqrt(var.clamp_min(1e-6))

    def _amplitude_temporal_inputs(self, voxel_features, voxel_num_points=None, valid_mask=None):
        valid = self._valid_mask(voxel_features, voxel_num_points) if valid_mask is None else valid_mask
        timestamp = self._timestamp(voxel_features)
        current_eps = float(self.model_cfg.get('CURRENT_TIME_EPS', 1e-3))
        current_mask = (timestamp.abs() <= current_eps).type_as(voxel_features) * valid
        history_mask = (timestamp.abs() > current_eps).type_as(voxel_features) * valid

        current_has_points = (current_mask.sum(dim=1, keepdim=True) > 0).type_as(voxel_features)
        history_has_points = (history_mask.sum(dim=1, keepdim=True) > 0).type_as(voxel_features)
        current_mask = current_mask * current_has_points + valid * (1.0 - current_has_points)
        history_mask = history_mask * history_has_points + valid * (1.0 - history_has_points)

        amplitude_values = self._amplitude_values(voxel_features)
        current_mean = self._masked_mean(amplitude_values, current_mask)
        history_mean = self._masked_mean(amplitude_values, history_mask)
        history_std = self._masked_std(amplitude_values, history_mask)
        delta_current = (amplitude_values - current_mean).abs()
        delta_history = (amplitude_values - history_mean).abs()
        return torch.cat(
            [delta_current, delta_history, history_std.expand_as(amplitude_values)],
            dim=-1,
        )

    def _gate_inputs(self, voxel_features, voxel_num_points=None, valid_mask=None):
        return torch.cat(
            [
                self._base_gate_inputs(voxel_features),
                *self._radial_inputs(voxel_features),
                self._amplitude_temporal_inputs(
                    voxel_features, voxel_num_points=voxel_num_points, valid_mask=valid_mask
                ),
            ],
            dim=-1,
        )

    def _point_weight(self, voxel_features, voxel_num_points):
        gate = self.point_gate(self._gate_inputs(voxel_features, voxel_num_points))
        return 1.0 + self.gate_scale * (gate - 0.5) * 2.0

    def _gate_raw_features(self, voxel_features, point_weight):
        return torch.cat(
            [
                voxel_features[:, :, :self.GATE_START_IDX],
                voxel_features[:, :, self.GATE_START_IDX:] * point_weight,
            ],
            dim=-1,
        )

    def _apply_point_gate(self, voxel_features, voxel_num_points):
        padded = self.get_paddings_indicator(
            voxel_num_points, voxel_features.shape[1], axis=0
        ).unsqueeze(-1).type_as(voxel_features)
        return self._gate_raw_features(
            voxel_features, self._point_weight(voxel_features, voxel_num_points)
        ) * padded

    def forward_export_onnx(self, batch_dict):
        pfn_features = batch_dict['voxels_cart'] # 15
        gate_features = batch_dict['gate_features']
        raw_dim = len(self.point_feature_names) # 9
        assert raw_dim + 6 == pfn_features.shape[-1]
        assert 5+3+6 == gate_features.shape[-1]

        gate = self.point_gate(gate_features)
        assert self.gate_scale == 0.35
        # point_weight = 1.0 + self.gate_scale * (gate - 0.5) * 2.0
        point_weight = 0.7 * gate + 0.65
        features = pfn_features * (1.0 + self.gate_mask * (point_weight - 1.0))

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict

    def forward(self, batch_dict, **kwargs):
        if getattr(self, 'export_onnx', False) and 'voxels_cart' in batch_dict:
            return self.forward_export_onnx(batch_dict)

        voxel_features, voxel_num_points, coords = (
            batch_dict['voxels'],
            batch_dict['voxel_num_points'],
            batch_dict['voxel_coords'],
        )
        point_weight = self._point_weight(voxel_features, voxel_num_points)
        padded = self.get_paddings_indicator(
            voxel_num_points, voxel_features.shape[1], axis=0
        ).unsqueeze(-1).type_as(voxel_features)
        voxel_features = voxel_features * padded

        points_mean = (
            voxel_features[:, :, :3].sum(dim=1, keepdim=True)
            / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        )
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = (
            voxel_features[:, :, 0]
            - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        )
        f_center[:, :, 1] = (
            voxel_features[:, :, 1]
            - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        )
        f_center[:, :, 2] = (
            voxel_features[:, :, 2]
            - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        )

        gated_voxel_features = self._gate_raw_features(voxel_features, point_weight)
        if self.use_absolute_xyz:
            features = [gated_voxel_features, f_cluster, f_center]
        else:
            features = [gated_voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
