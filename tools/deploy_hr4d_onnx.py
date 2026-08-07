import argparse
import json
import os
from pathlib import Path

import _init_path  # noqa: F401
import numpy as np
import onnx
from onnxsim import simplify

try:
    import onnx_graphsurgeon as gs
except ModuleNotFoundError:
    gs = None

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None

    class _MissingNN:
        class Module:
            pass

        class ModuleList(list):
            pass

    nn = _MissingNN()


class _DeployPointFeatureEncoder:
    def __init__(self, point_encoding_cfg):
        self.used_feature_list = list(point_encoding_cfg.used_feature_list)
        self.src_feature_list = list(point_encoding_cfg.src_feature_list)

    @property
    def num_point_features(self):
        return len(self.used_feature_list)


class _DeployDataset:
    """Minimal dataset contract needed by Detector3DTemplate.build_networks."""

    def __init__(self, dataset_cfg, class_names):
        self.dataset_cfg = dataset_cfg
        self.class_names = class_names
        self.point_cloud_range = np.asarray(dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = _DeployPointFeatureEncoder(dataset_cfg.POINT_FEATURE_ENCODING)
        voxel_cfg = _find_voxel_processor_cfg(dataset_cfg)
        self.voxel_size = np.asarray(voxel_cfg.VOXEL_SIZE, dtype=np.float32)
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
        ).astype(np.int64)
        self.depth_downsample_factor = None


class HR4DCenterPointExportWrapper(nn.Module):
    """Export current HR4D deployment graph: external VFE features -> pre-NMS heads."""

    def __init__(self, model):
        super().__init__()
        self.vfe = model.vfe
        self.dense_head = model.dense_head
        if not hasattr(self.vfe, 'forward_export_onnx'):
            raise RuntimeError('VFE must expose forward_export_onnx for deploy export')
        self.vfe.export_onnx = True

        if not bool(getattr(self.vfe, 'use_absolute_xyz', True)):
            raise RuntimeError('current deploy export requires USE_ABSLOTE_XYZ=True')

        vfe_index = None
        dense_index = None
        for idx, module in enumerate(model.module_list):
            if module is self.vfe:
                vfe_index = idx
            if module is self.dense_head:
                dense_index = idx
        if vfe_index is None:
            raise RuntimeError('vfe is not present in model.module_list')
        if dense_index is None:
            raise RuntimeError('dense_head is not present in model.module_list')
        if dense_index <= vfe_index:
            raise RuntimeError('dense_head must appear after vfe in model.module_list')
        self.post_vfe_modules = nn.ModuleList(model.module_list[vfe_index + 1:dense_index])

    def forward(self, voxels_cart, gate_features, voxel_idxs_cart):
        batch_dict = {
            'voxels_cart': voxels_cart,
            'gate_features': gate_features,
            'voxel_coords': voxel_idxs_cart,
            'batch_size': 1,
        }
        batch_dict = self.vfe(batch_dict)
        batch_dict['voxel_features'] = batch_dict['pillar_features']
        for module in self.post_vfe_modules:
            batch_dict = module(batch_dict)

        x = self.dense_head.shared_conv(batch_dict['spatial_features_2d'])
        outputs = []
        for head in self.dense_head.heads_list:
            pred_dict = head(x)
            for name in self.dense_head.separate_head_cfg.HEAD_ORDER:
                outputs.append(pred_dict[name])
            outputs.append(pred_dict['hm'])
        return tuple(outputs)


def _find_voxel_processor_cfg(dataset_cfg):
    for processor_cfg in dataset_cfg.DATA_PROCESSOR:
        if processor_cfg.NAME in {'transform_points_to_voxels', 'transform_points_to_voxels_placeholder'}:
            return processor_cfg
    raise KeyError('DATA_CONFIG.DATA_PROCESSOR must contain transform_points_to_voxels')


def _shape_defaults(model_cfg):
    voxel_cfg = _find_voxel_processor_cfg(model_cfg.DATA_CONFIG)
    max_voxels_cfg = voxel_cfg.MAX_NUMBER_OF_VOXELS
    if isinstance(max_voxels_cfg, dict):
        max_voxels = int(max_voxels_cfg.get('test', max_voxels_cfg.get('train')))
    else:
        max_voxels = int(max_voxels_cfg)
    max_points = int(voxel_cfg.MAX_POINTS_PER_VOXEL)
    return max_voxels, max_points


def _output_names(dense_head):
    names = []
    for head_idx, _ in enumerate(dense_head.heads_list):
        prefix = 'group_%d' % head_idx
        for name in dense_head.separate_head_cfg.HEAD_ORDER:
            names.append('%s_%s' % (prefix, name))
        names.append('%s_hm' % prefix)
    return names


def _pfn_input_dim(vfe):
    if not hasattr(vfe, 'pfn_layers') or len(vfe.pfn_layers) == 0:
        raise RuntimeError('deploy export requires a PillarVFE-style module with pfn_layers')
    first_pfn = vfe.pfn_layers[0]
    linear = getattr(first_pfn, 'linear', None)
    if linear is None or not hasattr(linear, 'in_features'):
        raise RuntimeError('deploy export currently supports Linear-based PFNLayer only')
    return int(linear.in_features)


def _point_gate_input_dim(vfe):
    point_gate = getattr(vfe, 'point_gate', None)
    if point_gate is None:
        raise RuntimeError('deploy export requires vfe.point_gate')
    for module in point_gate:
        if hasattr(module, 'in_features'):
            return int(module.in_features)
    raise RuntimeError('cannot infer point_gate input dim')


def _make_dummy_inputs(max_voxels, max_points, pfn_input_dim, gate_input_dim, grid_size, device):
    voxels_cart = torch.zeros((max_voxels, max_points, pfn_input_dim), dtype=torch.float32, device=device)
    voxels_cart[:, 0, 0] = 1.0
    gate_features = torch.zeros((max_voxels, max_points, gate_input_dim), dtype=torch.float32, device=device)
    voxel_idxs_cart = torch.zeros((max_voxels, 4), dtype=torch.int32, device=device)
    nx, ny = int(grid_size[0]), int(grid_size[1])
    linear = torch.arange(max_voxels, device=device, dtype=torch.int32)
    voxel_idxs_cart[:, 3] = linear % nx
    voxel_idxs_cart[:, 2] = (linear // nx) % ny
    return voxels_cart, gate_features, voxel_idxs_cart


def _write_manifest(path, args, model_cfg, input_shapes, output_names, pfn_input_dim, gate_input_dim, vfe):
    manifest = {
        'export_type': 'hr4d_centerpoint_gate_pfn_prenms_onnx',
        'cfg_file': str(Path(args.cfg_file).resolve()),
        'ckpt': str(Path(args.ckpt).resolve()) if args.ckpt else None,
        'onnx': str(Path(args.output_path).resolve()),
        'input_names': ['voxels_cart', 'gate_features', 'voxel_idxs_cart'],
        'input_shapes': input_shapes,
        'input_dtypes': {
            'voxels_cart': 'float32',
            'gate_features': 'float32',
            'voxel_idxs_cart': 'int32',
        },
        'output_names': output_names,
        'class_names': list(model_cfg.CLASS_NAMES),
        'point_feature_list': list(model_cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list),
        'point_cloud_range': list(model_cfg.DATA_CONFIG.POINT_CLOUD_RANGE),
        'voxel_size': list(_find_voxel_processor_cfg(model_cfg.DATA_CONFIG).VOXEL_SIZE),
        'pfn_input_dim': pfn_input_dim,
        'gate_input_dim': gate_input_dim,
        'gate_pfn_raw_feature_order': list(getattr(vfe, 'point_feature_names', [])),
        'postprocess_boundary': 'decode and NMS are intentionally outside this ONNX',
        'preprocess_contract': (
            'voxels_cart must contain the ungated PFN input features. gate_features must contain the '
            'externalized point_gate input features. Point padding masks and valid voxel counts are '
            'handled by preprocessing/deploy scatter; voxel_num is intentionally added only in final.onnx.'
        ),
    }
    Path(path).write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def _bev_shape(model_cfg):
    voxel_cfg = _find_voxel_processor_cfg(model_cfg.DATA_CONFIG)
    point_cloud_range = np.asarray(model_cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
    voxel_size = np.asarray(voxel_cfg.VOXEL_SIZE, dtype=np.float32)
    nx = int(round((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]))
    ny = int(round((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]))
    return ny, nx


def simplify_onnx(input_path, output_path, input_shapes, logger):
    model = onnx.load(input_path)
    model_simp, check = simplify(model, overwrite_input_shapes=input_shapes)
    if not check:
        raise RuntimeError('Simplified ONNX model could not be validated')
    onnx.checker.check_model(model_simp)
    onnx.save(model_simp, output_path)
    logger.info('[PASS] simplify ONNX exported to %s', output_path)
    return model_simp


def _find_node(graph, op, name_contains=None):
    for node in graph.nodes:
        if node.op != op:
            continue
        if name_contains and name_contains not in node.name:
            continue
        return node
    raise KeyError('cannot find node op=%s name_contains=%s' % (op, name_contains))


def _single_consumer(node, op=None):
    if not node.outputs or not node.outputs[0].outputs:
        raise KeyError('node %s has no consumer' % node.name)
    consumer = node.outputs[0].outputs[0]
    if op and consumer.op != op:
        raise KeyError('node %s consumer is %s, expected %s' % (node.name, consumer.op, op))
    return consumer


def _fold_reducemax_squeeze(graph, tensor_name, logger):
    tensors = graph.tensors()
    squeeze_output = tensors.get(tensor_name)
    if squeeze_output is None or not squeeze_output.inputs:
        return None

    squeeze = squeeze_output.inputs[0]
    if squeeze.op != 'Squeeze' or not squeeze.inputs or not squeeze.inputs[0].inputs:
        return squeeze_output

    reduce_max = squeeze.inputs[0].inputs[0]
    if reduce_max.op != 'ReduceMax':
        return squeeze_output

    keepdims = reduce_max.attrs.get('keepdims', 1)
    if keepdims not in (1, True):
        return squeeze_output

    reduce_output = squeeze.inputs[0]
    reduce_max.attrs['keepdims'] = 0
    reduce_output.shape = list(squeeze_output.shape or reduce_output.shape or [])
    reduce_output.dtype = squeeze_output.dtype or reduce_output.dtype

    for consumer in list(squeeze_output.outputs):
        consumer.inputs = [reduce_output if tensor is squeeze_output else tensor for tensor in consumer.inputs]
    for idx, output in enumerate(graph.outputs):
        if output is squeeze_output:
            graph.outputs[idx] = reduce_output

    squeeze.outputs.clear()
    logger.info('folded %s into %s with keepdims=0', squeeze.name, reduce_max.name)
    return reduce_output


def process_onnx(input_onnx, model_cfg, output_path, logger):
    if gs is None:
        raise ModuleNotFoundError('onnx_graphsurgeon is required for process_onnx')

    graph = gs.import_onnx(input_onnx)
    graph.cleanup().toposort()
    logger.info("process graph: '%s'", graph.name)

    pfn_output = _fold_reducemax_squeeze(graph, '/vfe/Squeeze_output_0', logger)
    graph.cleanup().toposort()
    tensors = graph.tensors()
    if pfn_output is None:
        pfn_output = tensors.get('/vfe/Squeeze_output_0')
    if pfn_output is None:
        pfn_output = tensors.get('/vfe/pfn_layers.0/ReduceMax_output_0')
    if pfn_output is None:
        raise KeyError('cannot find PFN output tensor for scatter plugin input')
    if 'voxel_idxs_cart' not in tensors:
        raise KeyError('cannot find voxel_idxs_cart input tensor')

    scatter = _find_node(graph, 'ScatterND', name_contains='post_vfe_modules.0')
    unsqueeze = _single_consumer(scatter, 'Unsqueeze')
    reshape = _single_consumer(unsqueeze, 'Reshape')
    plugin_output = reshape.outputs[0]

    pfn_dim = _pfn_input_dim_from_graph(pfn_output)
    ny, nx = _bev_shape(model_cfg)
    plugin_output.shape = [1, pfn_dim, ny, nx]
    plugin_output.dtype = np.float32

    voxel_num = gs.Variable(name='voxel_num', dtype=np.int32, shape=[1])
    graph.inputs.append(voxel_num)

    plugin_output.inputs.clear()
    graph.layer(
        name='PPScatterPlugin',
        op='PPScatterPlugin',
        inputs=[pfn_output, tensors['voxel_idxs_cart'], voxel_num],
        outputs=[plugin_output],
        attrs={'dense_shape': np.array([ny, nx], dtype=np.int32)},
    )

    graph.cleanup().toposort()
    processed = gs.export_onnx(graph)
    processed = onnx.shape_inference.infer_shapes(processed)
    onnx.save(processed, output_path)
    logger.info('[PASS] final ONNX exported to %s', output_path)
    return processed


def _pfn_input_dim_from_graph(pillar_tensor):
    shape = list(pillar_tensor.shape or [])
    if len(shape) >= 2 and isinstance(shape[-1], int):
        return int(shape[-1])
    # HR4D exp27 currently uses one PFN layer with 64 output channels.
    return 64


def parse_args():
    parser = argparse.ArgumentParser(description='Export and process HR4D current deploy ONNX')
    parser.add_argument('--cfg_file', required=True, help='Training YAML used to reconstruct the model')
    parser.add_argument('--ckpt', required=True, help='Checkpoint .pth')
    parser.add_argument('--output_path', required=True, help='Output ONNX path')
    parser.add_argument('--simplified_path', default=None, help='Optional simplified ONNX path')
    parser.add_argument('--final_path', default=None, help='Optional final processed ONNX path')
    parser.add_argument('--max_voxels', type=int, default=None, help='Override max voxel/pillar count')
    parser.add_argument('--max_points', type=int, default=None, help='Override max points per voxel')
    parser.add_argument('--opset', type=int, default=11)
    parser.add_argument('--cpu', action='store_true', help='Export on CPU instead of CUDA')
    parser.add_argument(
        '--keep_initializers_as_inputs',
        action='store_true',
        help='Expose weights as ONNX graph inputs for legacy consumers',
    )
    parser.add_argument('--manifest_path', default=None, help='Optional JSON manifest path')
    return parser.parse_args()


def main():
    args = parse_args()
    if torch is None:
        raise ModuleNotFoundError('torch is required for export; run this script in the HR4D CUDA/PyTorch container')

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.utils import common_utils

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()

    default_max_voxels, default_max_points = _shape_defaults(cfg)
    max_voxels = args.max_voxels or default_max_voxels
    max_points = args.max_points or default_max_points
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    dataset = _DeployDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=(device.type == 'cpu'))
    model.to(device).eval()

    pfn_input_dim = _pfn_input_dim(model.vfe)
    gate_input_dim = _point_gate_input_dim(model.vfe)
    wrapper = HR4DCenterPointExportWrapper(model).to(device).eval()
    dummy_inputs = _make_dummy_inputs(max_voxels, max_points, pfn_input_dim, gate_input_dim, dataset.grid_size, device)
    output_names = _output_names(model.dense_head)
    input_names = ['voxels_cart', 'gate_features', 'voxel_idxs_cart']
    input_shapes = {
        'voxels_cart': [max_voxels, max_points, pfn_input_dim],
        'gate_features': [max_voxels, max_points, gate_input_dim],
        'voxel_idxs_cart': [max_voxels, 4],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with torch.no_grad():
        test_outputs = wrapper(*dummy_inputs)
        for name, tensor in zip(output_names, test_outputs):
            logger.info('output %s shape=%s dtype=%s', name, tuple(tensor.shape), tensor.dtype)
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            args.output_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            keep_initializers_as_inputs=args.keep_initializers_as_inputs,
            input_names=input_names,
            output_names=output_names,
        )
    logger.info('ONNX exported to %s', args.output_path)

    manifest_path = args.manifest_path or os.path.splitext(args.output_path)[0] + '.manifest.json'
    _write_manifest(manifest_path, args, cfg, input_shapes, output_names, pfn_input_dim, gate_input_dim, model.vfe)
    logger.info('Export manifest written to %s', manifest_path)

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    simplified_path = args.simplified_path or os.path.join(output_dir, 'raw_simp.onnx')
    final_path = args.final_path or os.path.join(output_dir, 'final.onnx')
    simplified = simplify_onnx(
        args.output_path,
        simplified_path,
        {
            'voxels_cart': [max_voxels, max_points, pfn_input_dim],
            'gate_features': [max_voxels, max_points, gate_input_dim],
            'voxel_idxs_cart': [max_voxels, 4],
        },
        logger,
    )
    process_onnx(simplified, cfg, final_path, logger)


if __name__ == '__main__':
    main()
