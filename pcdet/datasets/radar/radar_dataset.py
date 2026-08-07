import copy
import io
import time
from pathlib import Path

import numpy as np
from easydict import EasyDict

from ...config import cfg
from ...utils import common_utils
from ...utils.storage import FileStorage, TOSStorage, build_storage
from ..dataset import DatasetTemplate


class RadarDataset(DatasetTemplate):
    """OpenPCDet dataset wrapper for the 4D radar info PKL format.

    Expected info keys per frame:
        radars: sensor-name -> radar_path, timestamp, radar2body, imu2world
        lidars: sensor-name -> lidar_path, lidar2body
        imu: imu2body
        annos: names, boxes_3d, optional vels and num_pts
        sweeps: previous radar frames with timestamp and imu2world

    Points from all sweeps are expressed in the current vehicle body frame.
    LiDAR points are also expressed in the current vehicle body frame.
    Ground-truth boxes already use that frame and are not transformed here.
    """

    DEFAULT_PCD_TYPE_MAP = {
        ('F', 4): '<f4',
        ('F', 8): '<f8',
        ('U', 1): 'u1',
        ('U', 2): '<u2',
        ('U', 4): '<u4',
        ('I', 1): 'i1',
        ('I', 2): '<i2',
        ('I', 4): '<i4',
    }

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        if root_path is None:
            root_path = dataset_cfg.DATA_PATH
        storage = build_storage(
            dataset_cfg=dataset_cfg, root_path=root_path,
            project_root=cfg.ROOT_DIR, logger=logger
        )
        storage_cfg = dataset_cfg.get('STORAGE', {})
        default_info_type = 'file' if storage.type == 'tos' else storage.type
        info_storage_type = str(storage_cfg.get('INFO_TYPE', default_info_type)).lower()
        pkl_path = dataset_cfg.get('PKL_PATH', root_path)
        if info_storage_type == 'file':
            self.info_storage = FileStorage(pkl_path, project_root=cfg.ROOT_DIR)
        elif info_storage_type == 'tos':
            self.info_storage = TOSStorage(pkl_path, storage_cfg=storage_cfg, logger=logger)
        else:
            raise ValueError('Unsupported info storage type: %s' % info_storage_type)

        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training,
            root_path=storage.root_path, logger=logger, storage=storage
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.infos = []
        self._bad_pose_log_count = 0
        self._bad_pose_log_limit = 20
        self.include_data(self.mode)

    def include_data(self, mode):
        self.logger.info('Loading Radar dataset.')
        infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_desc = self.info_storage.describe(info_path)
            self.logger.info('Checking Radar info path: %s', info_desc)
            if not self.info_storage.exists(info_path):
                self.logger.warning('Radar info path does not exist: %s', info_desc)
                continue
            start_time = time.time()
            self.logger.info('Loading Radar info file: %s', info_desc)
            loaded_infos = self.info_storage.load_pickle(info_path)
            infos.extend(loaded_infos)
            self.logger.info(
                'Loaded %d Radar infos from %s in %.1fs',
                len(loaded_infos), info_desc, time.time() - start_time
            )

        self.logger.info('Loaded %d Radar infos.', len(infos))

        if self.dataset_cfg.get('FILTER_EMPTY_FILES', True):
            filter_start = time.time()
            num_before = len(infos)
            missing_stats = {
                'current': 0,
            }
            filtered_infos = []
            for info in infos:
                keep, reason = self._has_required_input_files(info)
                if keep:
                    filtered_infos.append(info)
                else:
                    missing_stats[reason] = missing_stats.get(reason, 0) + 1
            infos = filtered_infos
            self.logger.info(
                'Filtered %d Radar infos without required files '
                '(missing current metadata=%d) in %.1fs.',
                num_before - len(infos),
                missing_stats['current'],
                time.time() - filter_start,
            )

        self.infos.extend(infos)
        self.logger.info('Total samples for Radar dataset: %d' % len(self.infos))

    def _radar_sensor(self):
        return self.dataset_cfg.get('RADAR_SENSOR', 'RADAR_FRONT')

    def _lidar_sensor(self):
        return self.dataset_cfg.get('LIDAR_SENSOR', 'LIDAR_FRONT_2')

    def _point_source(self):
        return str(self.dataset_cfg.get('POINT_SOURCE', 'radar')).lower()

    def _resolve_path(self, relative_path):
        if hasattr(self.storage, 'resolve_path'):
            return self.storage.resolve_path(relative_path)
        return Path(self.storage.describe(relative_path))

    @staticmethod
    def _is_valid_transform(value):
        if value is None:
            return False, 'None'
        try:
            matrix = np.asarray(value, dtype=np.float64)
        except Exception as exc:
            return False, 'convert_error:%s' % exc
        if matrix.shape != (4, 4):
            return False, 'shape:%s' % (matrix.shape,)
        if not np.isfinite(matrix).all():
            return False, 'non_finite'
        return True, None

    def _log_invalid_pose(self, info, location, field, detail):
        if self._bad_pose_log_count >= self._bad_pose_log_limit:
            return
        self._bad_pose_log_count += 1
        self.logger.warning(
            'Invalid Radar pose: sequence_id=%s frame_id=%s '
            'location=%s field=%s detail=%s',
            info.get('sequence_id'), info.get('frame_id'), location, field, detail
        )

    def _check_pose_field(self, info, location, field, value, reason):
        valid, detail = self._is_valid_transform(value)
        if valid:
            return True, None
        self._log_invalid_pose(info, location, field, detail)
        return False, reason

    def _has_required_radar_files(self, info):
        sensor_name = self._radar_sensor()
        sensor_info = info.get('radars', {}).get(sensor_name)
        radar_path = sensor_info.get('radar_path') if sensor_info else None
        if not radar_path:
            return False, 'current'

        return True, None

    def _has_required_lidar_files(self, info):
        sensor_name = self._lidar_sensor()
        sensor_info = info.get('lidars', {}).get(sensor_name)
        lidar_path = sensor_info.get('lidar_path') if sensor_info else None
        if not lidar_path or not self.storage.exists(lidar_path):
            return False, 'current'
        return True, None

    def _has_required_input_files(self, info):
        point_source = self._point_source()
        if point_source == 'radar':
            return self._has_required_radar_files(info)
        if point_source == 'lidar':
            return self._has_required_lidar_files(info)
        raise ValueError('Unsupported POINT_SOURCE: %s' % point_source)

    @staticmethod
    def _pcd_header_to_dtype(header_lines):
        header = {}
        for line in header_lines:
            parts = line.strip().split()
            if not parts:
                continue
            key = parts[0]
            if key in {'FIELDS', 'SIZE', 'TYPE', 'COUNT'}:
                header[key] = parts[1:]
            elif key in {'WIDTH', 'HEIGHT', 'POINTS'}:
                header[key] = int(parts[1])
            elif key == 'DATA':
                header[key] = parts[1]

        fields = header['FIELDS']
        sizes = [int(x) for x in header['SIZE']]
        types = header['TYPE']
        counts = [int(x) for x in header.get('COUNT', ['1'] * len(fields))]
        dtype_fields = []
        for name, size, type_name, count in zip(fields, sizes, types, counts):
            np_type = RadarDataset.DEFAULT_PCD_TYPE_MAP[(type_name, size)]
            dtype_fields.append((name, np_type) if count == 1 else (name, np_type, (count,)))
        return np.dtype(dtype_fields), fields, header

    @staticmethod
    def read_pcd_binary(file_path, storage=None):
        payload = storage.read_bytes(file_path) if storage is not None else Path(file_path).read_bytes()
        with io.BytesIO(payload) as f:
            header_lines = []
            while True:
                line = f.readline()
                if not line:
                    raise ValueError('Invalid PCD file without DATA line: %s' % file_path)
                text = line.decode('utf-8', errors='replace').strip()
                header_lines.append(text)
                if text.startswith('DATA'):
                    break
            dtype, fields, header = RadarDataset._pcd_header_to_dtype(header_lines)
            if header['DATA'] != 'binary':
                raise NotImplementedError('Only binary PCD is supported: %s' % file_path)
            points = np.frombuffer(f.read(), dtype=dtype, count=header['POINTS'])
        return points, fields

    @staticmethod
    def _transform_points(points, transform):
        xyz_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
        )
        return (xyz_h @ transform.T)[:, :3]

    @staticmethod
    def _compensate_ego_doppler(
        doppler, los_xyz_current, canbus, body_to_current_body,
        speed_scale=1.0 / 3.6, compensation_sign=1.0,
    ):
        """Remove translational ego motion from measured radial velocity.

        Doppler is measured along the radar line of sight. The vehicle speed
        is expressed in the sweep body frame, then rotated into the current
        body frame before projection onto the line of sight.
        """
        compensated = np.asarray(doppler, dtype=np.float32).copy()
        if not canbus or canbus.get('vehiclespeed') is None:
            return compensated

        speed = float(canbus['vehiclespeed']) * float(speed_scale)
        if not np.isfinite(speed):
            return compensated
        direction = str(canbus.get('vehicledirection', 'FORWARD')).upper()
        if direction in {'BACKWARD', 'REVERSE'}:
            speed = -abs(speed)
        elif direction in {'FORWARD', 'DRIVE'}:
            speed = abs(speed)

        body_to_current_body = np.asarray(
            body_to_current_body, dtype=np.float32
        )
        ego_velocity_body = np.array([speed, 0.0, 0.0], dtype=np.float32)
        ego_velocity_current = (
            ego_velocity_body @ body_to_current_body[:3, :3].T
        )

        los_xy = np.asarray(los_xyz_current, dtype=np.float32)[:, :2]
        ranges = np.linalg.norm(los_xy, axis=1)
        valid = ranges > 1e-6
        unit_los = np.zeros_like(los_xy)
        unit_los[valid] = los_xy[valid] / ranges[valid, None]
        ego_radial_velocity = unit_los @ ego_velocity_current[:2]
        compensated += float(compensation_sign) * ego_radial_velocity
        return compensated

    def _load_one_radar(
        self, radar_info, ref_timestamp, radar_to_current_body,
        body_to_current_body=None,
    ):
        pcd_path = radar_info['radar_path']
        if not self.storage.exists(pcd_path):
            return None

        pcd_points, fields = self.read_pcd_binary(pcd_path, storage=self.storage)
        feature_names = list(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        points = np.zeros((pcd_points.shape[0], len(feature_names)), dtype=np.float32)

        # Super-resolution PCDs use SR_x/SR_y/SR_z for generated coordinates,
        # while raw radar PCDs use x/y/z.  Keep the dataset interface in the
        # canonical x/y/z frame and accept either representation here.
        coordinate_fields = {
            axis: axis if axis in pcd_points.dtype.names else 'SR_' + axis
            for axis in ('x', 'y', 'z')
        }
        missing_coordinates = [
            field for field in coordinate_fields.values()
            if field not in pcd_points.dtype.names
        ]
        if missing_coordinates:
            raise ValueError(
                'Radar PCD %s is missing coordinate fields x/y/z or SR_x/SR_y/SR_z; '
                'missing=%s fields=%s' % (pcd_path, missing_coordinates, fields)
            )
        xyz = np.stack(
            [pcd_points[coordinate_fields[axis]] for axis in ('x', 'y', 'z')],
            axis=1,
        ).astype(np.float32)
        radar_to_current_body = np.asarray(radar_to_current_body, dtype=np.float32)
        xyz_body = self._transform_points(xyz, radar_to_current_body)

        ego_compensated_doppler = None
        if (
            'ego_comp_doppler' in feature_names
            and 'doppler' in pcd_points.dtype.names
        ):
            if body_to_current_body is None:
                body_to_current_body = np.eye(4, dtype=np.float32)
            los_xyz_current = xyz @ radar_to_current_body[:3, :3].T
            ego_compensated_doppler = self._compensate_ego_doppler(
                pcd_points['doppler'], los_xyz_current,
                radar_info.get('canbus', {}), body_to_current_body,
                speed_scale=float(self.dataset_cfg.get('EGO_SPEED_SCALE', 1.0 / 3.6)),
                compensation_sign=float(
                    self.dataset_cfg.get('EGO_DOPPLER_SIGN', 1.0)
                ),
            )

        velocity_body = None
        if {'Vx', 'Vy'}.issubset(pcd_points.dtype.names):
            velocity = np.stack([
                pcd_points['Vx'], pcd_points['Vy'],
                np.zeros(pcd_points.shape[0], dtype=np.float32)
            ], axis=1).astype(np.float32)
            velocity_body = velocity @ radar_to_current_body[:3, :3].T

        for i, name in enumerate(feature_names):
            if name == 'x':
                points[:, i] = xyz_body[:, 0]
            elif name == 'y':
                points[:, i] = xyz_body[:, 1]
            elif name == 'z':
                points[:, i] = xyz_body[:, 2]
            elif name == 'timestamp':
                points[:, i] = float(ref_timestamp - radar_info.get('timestamp', ref_timestamp))
            elif name == 'ego_comp_doppler' and ego_compensated_doppler is not None:
                points[:, i] = ego_compensated_doppler
            elif name == 'Vx' and velocity_body is not None:
                points[:, i] = velocity_body[:, 0]
            elif name == 'Vy' and velocity_body is not None:
                points[:, i] = velocity_body[:, 1]
            elif name in pcd_points.dtype.names:
                points[:, i] = pcd_points[name].astype(np.float32)

        return points

    def get_radar_points(self, info):
        sensor_name = self._radar_sensor()
        sensor_info = info.get('radars', {}).get(sensor_name)
        if sensor_info is None:
            return None

        points = []
        ref_timestamp = sensor_info.get('timestamp', info.get('timestamp', 0.0))
        keep, _ = self._check_pose_field(
            info, 'radars.%s' % sensor_name, 'radar2body',
            sensor_info.get('radar2body'), 'pose_current'
        )
        if not keep:
            return None
        keep, _ = self._check_pose_field(
            info, 'radars.%s' % sensor_name, 'imu2world',
            sensor_info.get('imu2world'), 'pose_current'
        )
        if not keep:
            return None
        radar2body = np.asarray(sensor_info['radar2body'], dtype=np.float64)
        current_points = self._load_one_radar(sensor_info, ref_timestamp, radar2body)
        if current_points is None:
            return None
        points.append(current_points)

        num_history = max(int(self.dataset_cfg.get('MAX_SWEEPS', 1)) - 1, 0)
        if num_history > 0:
            keep, _ = self._check_pose_field(
                info, 'imu', 'imu2body', info.get('imu', {}).get('imu2body'), 'pose_imu'
            )
            if not keep:
                return None
            imu2body = np.asarray(info['imu']['imu2body'], dtype=np.float64)
            body2imu = np.linalg.inv(imu2body)
            radar2imu = body2imu @ radar2body
            current_imu2world = np.asarray(sensor_info['imu2world'], dtype=np.float64)
            current_body2world = current_imu2world @ body2imu
            world2current_body = np.linalg.inv(current_body2world)

            for sweep_idx, sweep_info in enumerate(info.get('sweeps', {}).get(sensor_name, [])[:num_history]):
                sweep_path = sweep_info.get('radar_path')
                if not sweep_path or not self.storage.exists(sweep_path):
                    continue
                keep, _ = self._check_pose_field(
                    info, 'sweeps.%s[%d]' % (sensor_name, sweep_idx),
                    'imu2world', sweep_info.get('imu2world'), 'pose_sweep'
                )
                if not keep:
                    continue
                sweep_imu2world = np.asarray(sweep_info['imu2world'], dtype=np.float64)
                sweep_body2world = sweep_imu2world @ body2imu
                sweep_body2current_body = world2current_body @ sweep_body2world
                sweep_radar2world = sweep_imu2world @ radar2imu
                sweep_radar2current_body = world2current_body @ sweep_radar2world
                sweep_points = self._load_one_radar(
                    sweep_info, ref_timestamp, sweep_radar2current_body,
                    sweep_body2current_body,
                )
                if sweep_points is not None:
                    points.append(sweep_points)

        if not points:
            return np.zeros((0, len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)), dtype=np.float32)
        return np.concatenate(points, axis=0)

    def _load_one_lidar(self, lidar_info):
        pcd_path = lidar_info['lidar_path']
        if not self.storage.exists(pcd_path):
            return None

        pcd_points, fields = self.read_pcd_binary(pcd_path, storage=self.storage)
        feature_names = list(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        points = np.zeros((pcd_points.shape[0], len(feature_names)), dtype=np.float32)

        xyz = np.stack([pcd_points['x'], pcd_points['y'], pcd_points['z']], axis=1).astype(np.float32)
        lidar_to_body = np.asarray(lidar_info['lidar2body'], dtype=np.float32)
        xyz_body = self._transform_points(xyz, lidar_to_body)

        for i, name in enumerate(feature_names):
            if name == 'x':
                points[:, i] = xyz_body[:, 0]
            elif name == 'y':
                points[:, i] = xyz_body[:, 1]
            elif name == 'z':
                points[:, i] = xyz_body[:, 2]
            elif name in pcd_points.dtype.names:
                points[:, i] = pcd_points[name].astype(np.float32)

        return points

    def get_lidar_points(self, info):
        sensor_name = self._lidar_sensor()
        sensor_info = info.get('lidars', {}).get(sensor_name)
        if sensor_info is None:
            return np.zeros(
                (0, len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)), dtype=np.float32
            )
        points = self._load_one_lidar(sensor_info)
        if points is None:
            return np.zeros((0, len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)), dtype=np.float32)
        return points

    def _map_names(self, raw_names):
        class_mapping = self.dataset_cfg.get('CLASS_MAPPING', {})
        mapped_names = [class_mapping.get(str(name), str(name)) for name in raw_names]
        return np.array(mapped_names)

    def _num_points_mask(self, annos, num_boxes, for_eval=False):
        min_points_key = 'MIN_GT_POINTS_EVAL' if for_eval else 'MIN_GT_POINTS_TRAIN'
        min_points = int(self.dataset_cfg.get(min_points_key, self.dataset_cfg.get('MIN_GT_POINTS', 0)))
        sensor_name = self.dataset_cfg.get('NUM_POINTS_SENSOR', self._lidar_sensor() if self._point_source() == 'lidar' else self._radar_sensor())
        sensor_num_points = annos.get('num_pts', {}).get(sensor_name)
        if sensor_num_points is None:
            return np.ones(num_boxes, dtype=np.bool_)

        sensor_num_points = np.asarray(sensor_num_points).reshape(-1)
        if len(sensor_num_points) != num_boxes:
            raise ValueError(
                'num_pts[%s] has %d entries, but boxes_3d has %d'
                % (sensor_name, len(sensor_num_points), num_boxes)
            )
        return sensor_num_points >= min_points

    def get_annos(self, info, for_eval=False, return_mask=False):
        annos = info.get('annos', {})
        raw_names = annos.get('names', np.array([], dtype=str))
        gt_names = self._map_names(raw_names)
        gt_boxes = annos.get('boxes_3d', np.zeros((0, 7), dtype=np.float32)).astype(np.float32)

        if self.dataset_cfg.get('INCLUDE_GT_SPEED', False):
            gt_vels = annos.get('vels')
            if gt_vels is not None and len(gt_vels) == len(gt_boxes):
                gt_vels = np.nan_to_num(gt_vels.astype(np.float32), nan=0.0)
                gt_boxes = np.concatenate([gt_boxes, gt_vels], axis=1)

        selected = self._num_points_mask(annos, len(gt_boxes), for_eval=for_eval)
        if return_mask:
            return gt_names[selected], gt_boxes[selected], selected
        return gt_names[selected], gt_boxes[selected]

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def _sample_retry_limit(self):
        return max(1, min(
            len(self.infos),
            int(self.dataset_cfg.get('MAX_SAMPLE_RETRY', 50))
        ))

    def _next_retry_index(self, index):
        if self.training:
            return np.random.randint(self.__len__())
        return (index + 1) % self.__len__()

    def _build_input_dict(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        point_source = self._point_source()
        if point_source == 'radar':
            points = self.get_radar_points(info)
        elif point_source == 'lidar':
            points = self.get_lidar_points(info)
        else:
            raise ValueError('Unsupported POINT_SOURCE: %s' % point_source)
        if points is None or len(points) == 0:
            return None
        input_dict = {
            'frame_id': info.get('frame_id', str(index)),
            'metadata': {
                'sequence_id': info.get('sequence_id'),
                'timestamp': info.get('timestamp'),
            },
            'points': points,
        }

        if 'annos' in info:
            gt_names, gt_boxes = self.get_annos(info, for_eval=not self.training)
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes,
            })

        return input_dict

    def __getitem__(self, index):
        retry_limit = self._sample_retry_limit()
        last_index = index
        for _ in range(retry_limit):
            input_dict = self._build_input_dict(index)
            if input_dict is not None:
                return self.prepare_data(data_dict=input_dict)
            last_index = index
            index = self._next_retry_index(index)

        raise RuntimeError(
            'Failed to load a valid Radar sample after %d retries, last_index=%s'
            % (retry_limit, last_index)
        )

    def _build_eval_gt_annos(self, class_names):
        eval_gt_annos = []
        for info_idx, info in enumerate(self.infos):
            annos = info.get('annos', {})
            gt_names, gt_boxes, gt_selected = self.get_annos(
                info, for_eval=True, return_mask=True
            )
            class_selected = common_utils.keep_arrays_by_name(gt_names, class_names)
            gt_anno = {
                'frame_id': info.get('frame_id', str(info_idx)),
                'name': gt_names[class_selected],
                'boxes_3d': gt_boxes[class_selected],
            }

            for key in (
                'dynamic_gt_candidate',
                'speed_abs_xy_candidate',
                'gt_velocity_xy_from_radar_candidate',
            ):
                if key not in annos:
                    continue
                values = np.asarray(annos[key])
                if len(values) != len(gt_selected):
                    raise ValueError(
                        '%s has %d entries, but boxes_3d has %d'
                        % (key, len(values), len(gt_selected))
                    )
                gt_anno[key] = values[gt_selected][class_selected]

            eval_gt_annos.append(gt_anno)
        return eval_gt_annos

    @staticmethod
    def _anno_frame_id(anno):
        frame_id = anno.get('frame_id', None)
        if frame_id is None:
            return None
        return str(frame_id)

    def _align_eval_annos_by_frame_id(self, eval_gt_annos, eval_det_annos):
        gt_frame_ids = [self._anno_frame_id(anno) for anno in eval_gt_annos]
        det_frame_ids = [self._anno_frame_id(anno) for anno in eval_det_annos]
        if not all(gt_frame_ids) or not all(det_frame_ids):
            return eval_gt_annos, eval_det_annos

        gt_by_frame_id = {}
        for frame_id, anno in zip(gt_frame_ids, eval_gt_annos):
            if frame_id not in gt_by_frame_id:
                gt_by_frame_id[frame_id] = anno

        aligned_gt_annos, aligned_det_annos = [], []
        seen_det_frame_ids = set()
        skipped_duplicate = 0
        skipped_without_gt = 0
        for frame_id, anno in zip(det_frame_ids, eval_det_annos):
            if frame_id in seen_det_frame_ids:
                skipped_duplicate += 1
                continue
            seen_det_frame_ids.add(frame_id)
            gt_anno = gt_by_frame_id.get(frame_id)
            if gt_anno is None:
                skipped_without_gt += 1
                continue
            aligned_gt_annos.append(gt_anno)
            aligned_det_annos.append(anno)

        missing_pred = len(gt_by_frame_id) - len(aligned_gt_annos)
        if self.logger is not None and (
            skipped_duplicate or skipped_without_gt or missing_pred or len(aligned_gt_annos) != len(eval_gt_annos)
        ):
            self.logger.info(
                'Aligned evaluation by frame_id: gt=%d, pred=%d, matched=%d, '
                'missing_pred=%d, duplicate_pred=%d, pred_without_gt=%d',
                len(eval_gt_annos), len(eval_det_annos), len(aligned_gt_annos),
                missing_pred, skipped_duplicate, skipped_without_gt
            )
        return aligned_gt_annos, aligned_det_annos

    @staticmethod
    def _eval_metric_name(eval_metric):
        if isinstance(eval_metric, (dict, EasyDict)):
            return str(eval_metric.get('NAME', eval_metric.get('name', 'hr4d')))
        return str(eval_metric)

    def _hr4d_eval_config(self, eval_metric=None, sensor_type=None):
        from ..hr4d.hr4d_eval import HR4DEvalConfig

        metric_cfg = eval_metric if isinstance(eval_metric, (dict, EasyDict)) else {}
        if sensor_type is None:
            sensor_type = metric_cfg.get('HR4D_EVAL_SENSOR', None)
        if sensor_type is None:
            sensor_type = metric_cfg.get('SENSOR', None)
        if sensor_type is None:
            sensor_type = self.dataset_cfg.get('HR4D_EVAL_SENSOR', None)
        if sensor_type is None:
            sensor_type = self.dataset_cfg.get('POINT_SOURCE', 'RADAR')

        eval_range = metric_cfg.get('EVAL_RANGE', self.dataset_cfg.get('HR4D_EVAL_RANGE', [0.0, 200.0]))
        far_dynamic_vehicle = metric_cfg.get(
            'FAR_DYNAMIC_VEHICLE',
            self.dataset_cfg.get('HR4D_FAR_DYNAMIC_VEHICLE', [100.0, 200.0]),
        )
        return HR4DEvalConfig.from_ranges(
            sensor_type=sensor_type,
            eval_range=eval_range,
            far_dynamic_vehicle=far_dynamic_vehicle,
        )

    def _evaluate_hr4d(self, eval_gt_annos, eval_det_annos, class_names, eval_metric=None, sensor_type=None):
        from ..hr4d.hr4d_eval import get_evaluation_results

        return get_evaluation_results(
            gt_annos=eval_gt_annos,
            pred_annos=eval_det_annos,
            class_names=class_names,
            config=self._hr4d_eval_config(eval_metric=eval_metric, sensor_type=sensor_type),
        )

    def _evaluate_kitti(self, eval_gt_annos, eval_det_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval
        from ..kitti import kitti_utils

        map_name_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI
        kitti_gt_annos = [{
            'name': anno['name'].copy(),
            'gt_boxes_lidar': anno['boxes_3d'][:, :7],
        } for anno in eval_gt_annos]

        kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
        kitti_utils.transform_annotations_to_kitti_format(kitti_gt_annos, map_name_to_kitti=map_name_to_kitti)
        kitti_class_names = [map_name_to_kitti[x] for x in class_names]
        return kitti_eval.get_official_eval_result(
            gt_annos=kitti_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )

    def evaluation(self, det_annos, class_names, **kwargs):
        if not self.infos or 'annos' not in self.infos[0]:
            return 'No ground-truth boxes for evaluation', {}

        eval_metric = kwargs.get('eval_metric', self.dataset_cfg.get('EVAL_METRIC', 'hr4d'))
        eval_metric_name = self._eval_metric_name(eval_metric).lower()
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = self._build_eval_gt_annos(class_names)
        eval_gt_annos, eval_det_annos = self._align_eval_annos_by_frame_id(
            eval_gt_annos, eval_det_annos
        )

        if eval_metric_name == 'hr4d':
            return self._evaluate_hr4d(
                eval_gt_annos,
                eval_det_annos,
                class_names,
                eval_metric=eval_metric,
                sensor_type=kwargs.get('hr4d_eval_sensor', None),
            )
        if eval_metric_name == 'kitti':
            return self._evaluate_kitti(eval_gt_annos, eval_det_annos, class_names)

        raise NotImplementedError('Unsupported RadarDataset evaluation metric: %s' % eval_metric_name)
