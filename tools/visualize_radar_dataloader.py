import _init_path
import argparse
from itertools import islice
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from PIL import Image
except ImportError as error:
    raise ImportError(
        'Radar visualization requires matplotlib and Pillow. '
        'Install them in the OpenPCDet environment with: pip install matplotlib pillow'
    ) from error

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


CAMERA_VIEWS = (
    ('FW', 'CAMERA_FRONT_WIDE'),
    ('FN', 'CAMERA_FRONT_FAR'),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize RadarDataset camera images, single-frame BEV and four-frame BEV.'
    )
    parser.add_argument(
        '--cfg_file', default='cfgs/radar_models/second_radar.yaml',
        help='Model config path. Run this script from the OpenPCDet tools directory.'
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument('--index', type=int, default=0, help='Index in the filtered RadarDataset.')
    sample_group.add_argument('--frame_id', type=str, help='Select a sample by frame_id.')
    parser.add_argument('--workers', type=int, default=0, help='Dataloader worker count.')
    parser.add_argument(
        '--min_gt_points', type=int, default=None,
        help='Override DATA_CONFIG.MIN_GT_POINTS for both displayed GT and evaluation.'
    )
    parser.add_argument('--point_size', type=float, default=0.35)
    parser.add_argument(
        '--max_points', type=int, default=150000,
        help='Maximum displayed points per BEV panel; <=0 disables display downsampling.'
    )
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--dpi', type=int, default=160)
    parser.add_argument('--output', type=str, default=None, help='Output PNG path.')
    return parser.parse_args()


def resolve_sample_index(dataset, requested_index, frame_id):
    if frame_id is None:
        if requested_index < 0 or requested_index >= len(dataset):
            raise IndexError('index %d is outside dataset size %d' % (requested_index, len(dataset)))
        return requested_index

    for index, info in enumerate(dataset.infos):
        if str(info.get('frame_id')) == frame_id:
            return index
    raise KeyError('frame_id not found in filtered dataset: %s' % frame_id)


def get_batch_at(loader, index):
    batch = next(islice(loader, index, index + 1), None)
    if batch is None:
        raise IndexError('Unable to load dataloader sample at index %d' % index)
    return batch


def extract_batch_points(batch):
    points = np.asarray(batch['points'])
    if points.shape[1] < 5:
        raise ValueError('Expected collated points with batch index and radar features, got %s' % (points.shape,))

    points = points[points[:, 0] == 0][:, 1:]
    if len(points) == 0:
        raise ValueError('Dataloader returned an empty point cloud')
    return points


def extract_batch_gt(batch):
    gt_boxes = np.asarray(batch.get('gt_boxes', np.zeros((1, 0, 8), dtype=np.float32)))[0]
    if len(gt_boxes) == 0:
        return gt_boxes

    # The last column is the one-based class id added by DatasetTemplate.prepare_data.
    return gt_boxes[gt_boxes[:, -1] > 0]


def load_camera_image(dataset, info, camera_name):
    camera_info = info.get('cameras', {}).get(camera_name)
    if camera_info is None:
        return None, 'Missing camera metadata: %s' % camera_name

    image_path = dataset._resolve_path(camera_info['image_path'])
    if not image_path.exists():
        return None, 'Missing image: %s' % image_path

    with Image.open(image_path) as image:
        return np.asarray(image.convert('RGB')), str(image_path)


def downsample_points(points, max_points, rng):
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = rng.choice(len(points), max_points, replace=False)
    return points[indices]


def box_corners_bev(box):
    x, y, _, dx, dy, _, yaw = box[:7]
    local = np.array([
        [dx / 2, dy / 2],
        [dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
        [-dx / 2, dy / 2],
    ], dtype=np.float32)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return local @ rotation.T + np.array([x, y], dtype=np.float32)


def draw_gt_boxes(ax, gt_boxes, class_names):
    colors = ('#ff4d4f', '#22c55e', '#f59e0b', '#a855f7', '#06b6d4')
    for box in gt_boxes:
        class_id = int(box[-1])
        class_name = class_names[class_id - 1] if 0 < class_id <= len(class_names) else str(class_id)
        color = colors[(class_id - 1) % len(colors)]
        corners = box_corners_bev(box)
        closed = np.vstack([corners, corners[0]])

        # Plot lateral y horizontally and longitudinal x vertically, so vehicle front points upward.
        ax.plot(closed[:, 1], closed[:, 0], color=color, linewidth=1.25)
        x, y, _, dx, _, _, yaw = box[:7]
        front_x = x + np.cos(yaw) * dx / 2
        front_y = y + np.sin(yaw) * dx / 2
        ax.plot([y, front_y], [x, front_x], color=color, linewidth=1.25)
        ax.text(y, x, class_name, color=color, fontsize=6, ha='center', va='bottom')


def draw_bev(
        ax, points, gt_boxes, class_names, title, point_size,
        point_cloud_range, timestamp_index=None):
    if timestamp_index is not None:
        scatter = ax.scatter(
            points[:, 1], points[:, 0], c=points[:, timestamp_index], s=point_size,
            cmap='viridis', vmin=0.0,
            vmax=max(float(points[:, timestamp_index].max()), 1e-3),
            linewidths=0, alpha=0.8, rasterized=True
        )
        colorbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label('History age (s)', fontsize=7)
        colorbar.ax.tick_params(labelsize=6)
    else:
        ax.scatter(
            points[:, 1], points[:, 0], s=point_size, c='#2563eb',
            linewidths=0, alpha=0.75, rasterized=True
        )

    draw_gt_boxes(ax, gt_boxes, class_names)
    ax.scatter([0], [0], marker='^', s=35, c='black', zorder=5)
    ax.annotate('', xy=(0, 8), xytext=(0, 0), arrowprops={'arrowstyle': '->', 'color': 'black'})

    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    # FLU: +x is forward and +y is left. Reverse the displayed y axis so
    # positive lateral coordinates appear on the left side of the BEV.
    ax.set_xlim(y_max, y_min)
    ax.set_ylim(x_min, x_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.set_xlabel('Lateral y (m), left positive')
    ax.set_ylabel('Longitudinal x (m), forward positive')
    ax.set_title('%s\npoints=%d, GT=%d' % (title, len(points), len(gt_boxes)), fontsize=10)


def draw_camera(ax, image, short_name, camera_name, message):
    ax.set_xticks([])
    ax.set_yticks([])
    if image is None:
        ax.set_facecolor('#20242a')
        ax.text(0.5, 0.5, message, color='white', ha='center', va='center', wrap=True, transform=ax.transAxes)
    else:
        ax.imshow(image)
    ax.set_title('%s  %s' % (short_name, camera_name), fontsize=10, loc='left')


def main():
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.min_gt_points is not None:
        cfg.DATA_CONFIG.MIN_GT_POINTS = args.min_gt_points

    logger = common_utils.create_logger(rank=0)
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    sample_index = resolve_sample_index(dataset, args.index, args.frame_id)
    batch = get_batch_at(dataloader, sample_index)
    info = dataset.infos[sample_index]

    stacked_points = extract_batch_points(batch)
    used_features = list(cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list)
    if 'timestamp' not in used_features:
        raise KeyError('timestamp must be present in POINT_FEATURE_ENCODING.used_feature_list')
    timestamp_index = used_features.index('timestamp')
    single_frame_mask = np.isclose(stacked_points[:, timestamp_index], 0.0, atol=1e-4)
    single_points = stacked_points[single_frame_mask]
    gt_boxes = extract_batch_gt(batch)

    displayed_single = downsample_points(single_points, args.max_points, rng)
    displayed_stacked = downsample_points(stacked_points, args.max_points, rng)

    figure = plt.figure(figsize=(20, 9), constrained_layout=True)
    grid = GridSpec(2, 3, figure=figure, width_ratios=[1.2, 1.0, 1.0])
    for row, (short_name, camera_name) in enumerate(CAMERA_VIEWS):
        axis = figure.add_subplot(grid[row, 0])
        image, message = load_camera_image(dataset, info, camera_name)
        draw_camera(axis, image, short_name, camera_name, message)

    single_axis = figure.add_subplot(grid[:, 1])
    stacked_axis = figure.add_subplot(grid[:, 2])
    draw_bev(
        single_axis, displayed_single, gt_boxes, cfg.CLASS_NAMES,
        'Single-frame RADAR_FRONT + GT', args.point_size,
        cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    )
    draw_bev(
        stacked_axis, displayed_stacked, gt_boxes, cfg.CLASS_NAMES,
        'Current + previous 3 sweeps + GT', args.point_size,
        cfg.DATA_CONFIG.POINT_CLOUD_RANGE, timestamp_index=timestamp_index
    )

    frame_id = str(info.get('frame_id', sample_index))
    min_gt_points = int(cfg.DATA_CONFIG.get('MIN_GT_POINTS', 0))
    figure.suptitle(
        'Radar dataloader sample %d | frame_id=%s | MIN_GT_POINTS=%d'
        % (sample_index, frame_id, min_gt_points), fontsize=13
    )

    if args.output is None:
        output_path = Path('output/radar_visualization') / ('%04d_%s.png' % (sample_index, frame_id))
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print('Saved visualization to %s' % output_path.resolve())

    plt.close(figure)


if __name__ == '__main__':
    main()
