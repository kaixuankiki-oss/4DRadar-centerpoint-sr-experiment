"""Radar super-resolution model loading and inference helpers.

This module is the compatibility layer used by ``reconstructed_inference.py``.
The workspace currently contains the epoch-60 ZYNQ sparse U-Net and its
checkpoint.  ORIN and PTQ entry points are retained, but require their model
classes to be present in ``model.py``.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

import spconv.pytorch as spconv

from model import UNet_Generator_ZYNQ


PathLike = Union[str, os.PathLike[str]]
InferenceInput = Union[torch.Tensor, spconv.SparseConvTensor]
PointResult = Union[torch.Tensor, List[torch.Tensor]]


def _optional_model_class(name: str):
    """Return an optional class from model.py with a useful error message."""
    import model as model_module

    cls = getattr(model_module, name, None)
    if cls is None:
        raise ImportError(
            f"当前目录的 model.py 未提供 {name}。"
            "现有 model_epoch_60.pth 对应 UNet_Generator_ZYNQ，请使用 model_type='zynq'。"
        )
    return cls


def _checkpoint_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"不支持的 checkpoint 类型: {type(checkpoint).__name__}")

    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            state_dict = candidate
            break
    else:
        state_dict = checkpoint

    if not state_dict or not all(isinstance(key, str) for key in state_dict):
        raise ValueError("checkpoint 中没有找到有效的模型 state_dict")

    # DDP and torch.compile commonly add one or both of these prefixes.
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod.") :]
        cleaned[key] = value
    return cleaned


def load_model(
    checkpoint_path: PathLike,
    device: torch.device,
    model_type: str = "orin",
    is_super_resolution: bool = True,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
) -> nn.Module:
    """Load a trained radar super-resolution model in evaluation mode."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint 文件不存在: {checkpoint_path}")

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False")

    normalized_type = model_type.lower()
    if normalized_type == "zynq":
        model_cls = UNet_Generator_ZYNQ
    elif normalized_type == "orin":
        model_cls = _optional_model_class("UNet_Generator_ORIN")
    else:
        raise ValueError(f"不支持的模型类型: {model_type!r}，应为 'zynq' 或 'orin'")

    model = model_cls(is_super_resolution=is_super_resolution)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    # Store the effective inference configuration for diagnostics and wrappers.
    model._radar_sr_model_type = normalized_type  # type: ignore[attr-defined]
    model._radar_sr_use_amp = bool(use_amp)  # type: ignore[attr-defined]
    model._radar_sr_amp_dtype = amp_dtype  # type: ignore[attr-defined]

    print(f"模型已加载: {checkpoint_path}")
    print(
        f"模型类型: {normalized_type}, 超分辨率: {is_super_resolution}, "
        f"AMP: {use_amp} ({amp_dtype}), 设备: {device}"
    )
    return model


def load_quantized_model(
    checkpoint_path: PathLike,
    device: torch.device,
    is_super_resolution: bool = True,
) -> nn.Module:
    """Load the optional ZYNQ PTQ model when its class is available."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint 文件不存在: {checkpoint_path}")

    model_cls = _optional_model_class("UNet_Generator_ZYNQ_PTQ")
    model = model_cls(is_super_resolution=is_super_resolution)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=True)
    model = model.to(device)
    model.eval()
    print(f"量化模型已加载: {checkpoint_path}")
    return model


def extract_points_from_output(
    output: torch.Tensor,
    threshold: float = 0.5,
    return_probability: bool = False,
) -> PointResult:
    """Convert a ``[B,C,H,W]`` probability map to radar-bin coordinates.

    Returned coordinates use ``(RangeBin, AziBin, EleBin)`` order.  A single
    sample returns one tensor; a batch returns a list with one tensor per item.
    """
    if not torch.is_tensor(output):
        raise TypeError(f"output 必须是 torch.Tensor，实际为 {type(output).__name__}")
    if output.ndim == 3:
        output = output.unsqueeze(0)
    if output.ndim != 4:
        raise ValueError(f"output 形状应为 [B,C,H,W] 或 [C,H,W]，实际为 {tuple(output.shape)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold 必须位于 [0,1]，实际为 {threshold}")

    all_points: List[torch.Tensor] = []
    for prob_map in output:
        mask = prob_map > threshold
        indices = torch.nonzero(mask, as_tuple=False)  # (elevation, range, azimuth)
        if indices.numel() == 0:
            columns = 4 if return_probability else 3
            points = torch.empty((0, columns), device=output.device, dtype=torch.float32)
        else:
            coords = torch.stack(
                (indices[:, 1], indices[:, 2], indices[:, 0]), dim=1
            ).to(torch.float32)
            if return_probability:
                probabilities = prob_map[mask].to(torch.float32).unsqueeze(1)
                points = torch.cat((coords, probabilities), dim=1)
            else:
                points = coords
        all_points.append(points)

    return all_points[0] if len(all_points) == 1 else all_points


def _move_input(input_tensor: InferenceInput, device: torch.device) -> InferenceInput:
    if isinstance(input_tensor, torch.Tensor):
        return input_tensor.to(device)
    if isinstance(input_tensor, spconv.SparseConvTensor):
        if input_tensor.features.device == device and input_tensor.indices.device == device:
            return input_tensor
        # spconv 2.3 SparseConvTensor does not implement Tensor.to().  Rebuild
        # the input tensor on the requested device; inference inputs do not
        # need to preserve an existing indice cache.
        return spconv.SparseConvTensor(
            features=input_tensor.features.to(device),
            indices=input_tensor.indices.to(device),
            spatial_shape=input_tensor.spatial_shape,
            batch_size=input_tensor.batch_size,
        )
    raise TypeError(
        "input_tensor 必须是 torch.Tensor 或 spconv.SparseConvTensor，"
        f"实际为 {type(input_tensor).__name__}"
    )


def _forward_eval(model: nn.Module, input_tensor: InferenceInput):
    """Pass is_eval when supported by the model's forward signature."""
    signature = inspect.signature(model.forward)
    if "is_eval" in signature.parameters:
        return model(input_tensor, is_eval=True)
    return model(input_tensor)


def run_inference(
    model: nn.Module,
    input_tensor: InferenceInput,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    threshold: float = 0.5,
    model_type: str = "orin",
    return_probability: bool = False,
    return_prob_map: bool = False,
):
    """Run inference and extract occupied radar-bin coordinates."""
    device = torch.device(device)
    normalized_type = model_type.lower()
    if normalized_type not in {"zynq", "orin"}:
        raise ValueError(f"不支持的模型类型: {model_type!r}")
    if amp_dtype not in {"fp16", "bf16"}:
        raise ValueError("amp_dtype 必须为 'fp16' 或 'bf16'")

    input_tensor = _move_input(input_tensor, device)
    autocast_dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    amp_enabled = bool(use_amp and device.type == "cuda")

    with torch.inference_mode():
        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=autocast_dtype,
        ):
            output = _forward_eval(model, input_tensor)

    if isinstance(output, spconv.SparseConvTensor):
        output = output.dense()
    if not torch.is_tensor(output):
        raise TypeError(f"模型输出必须为 Tensor 或 SparseConvTensor，实际为 {type(output).__name__}")

    points = extract_points_from_output(
        output, threshold=threshold, return_probability=return_probability
    )
    return (points, output) if return_prob_map else points


def _pcd_numpy_dtype(fields, sizes, types, counts):
    type_map = {
        ("F", 4): "<f4",
        ("F", 8): "<f8",
        ("I", 1): "<i1",
        ("I", 2): "<i2",
        ("I", 4): "<i4",
        ("I", 8): "<i8",
        ("U", 1): "<u1",
        ("U", 2): "<u2",
        ("U", 4): "<u4",
        ("U", 8): "<u8",
    }
    dtype_fields = []
    for field, size, kind, count in zip(fields, sizes, types, counts):
        dtype = type_map.get((kind.upper(), size))
        if dtype is None:
            raise ValueError(f"不支持的 PCD 字段类型: {field} TYPE={kind} SIZE={size}")
        dtype_fields.append((field, dtype) if count == 1 else (field, dtype, (count,)))
    return np.dtype(dtype_fields)


def _read_binary_pcd(file_path: PathLike) -> np.ndarray:
    header: Dict[str, List[str]] = {}
    with open(file_path, "rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"PCD 头部缺少 DATA 行: {file_path}")
            line = raw.decode("utf-8").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                if len(parts) < 2 or parts[1].lower() != "binary":
                    raise ValueError("目前仅支持 DATA binary 格式的 PCD")
                break

        fields = header["FIELDS"]
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        points = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
        return np.fromfile(handle, dtype=_pcd_numpy_dtype(fields, sizes, types, counts), count=points)


def _field(data: np.ndarray, *aliases: str) -> np.ndarray:
    names = {name.lower(): name for name in (data.dtype.names or ())}
    for alias in aliases:
        actual = names.get(alias.lower())
        if actual is not None:
            return np.asarray(data[actual])
    raise ValueError(f"PCD 缺少字段，候选名称: {aliases}")


def load_single_pcd_file(
    file_path: PathLike,
    spatial_shape: List[int] = [512, 128],
    c_slices: int = 64,
    device: torch.device = torch.device("cuda"),
    add_offset: bool = True,
) -> spconv.SparseConvTensor:
    """Load one binary PCD into the sparse format used by the ZYNQ model."""
    data = _read_binary_pcd(file_path)
    range_bins = _field(data, "RangeBin", "range_bin", "h").astype(np.int32)
    azi_bins = _field(data, "AziBin", "azi_bin", "w").astype(np.int32)
    ele_bins = _field(data, "EleBin", "ele_bin", "c").astype(np.int32)
    rcs = _field(data, "RCS", "rcs").astype(np.float32)
    # The current radar_front_bottom PCD stores both raw ``doppler`` and the
    # ego-motion-compensated ``AbsV``.  EnhancedBinDataset's row[3] is the
    # compensated value (called AbsDoppler in older six-float files).
    doppler = _field(data, "AbsDoppler", "AbsV", "doppler").astype(np.float32)
    if add_offset:
        rcs = rcs + 70.0
        doppler = doppler + 70.0

    h_max, w_max = spatial_shape
    ele_count = c_slices // 2
    valid = (
        (range_bins >= 0)
        & (range_bins < h_max)
        & (azi_bins >= 0)
        & (azi_bins < w_max)
        & (ele_bins >= 0)
        & (ele_bins < ele_count)
    )
    range_bins, azi_bins, ele_bins = range_bins[valid], azi_bins[valid], ele_bins[valid]
    rcs, doppler = rcs[valid], doppler[valid]

    linear = range_bins * w_max + azi_bins
    unique_linear, inverse = np.unique(linear, return_inverse=True)
    features = np.zeros((len(unique_linear), c_slices), dtype=np.float32)
    features[inverse, 2 * ele_bins] = rcs
    features[inverse, 2 * ele_bins + 1] = doppler

    indices = np.zeros((len(unique_linear), 3), dtype=np.int32)
    indices[:, 1] = unique_linear // w_max
    indices[:, 2] = unique_linear % w_max
    sparse_tensor = spconv.SparseConvTensor(
        features=torch.from_numpy(features).to(device),
        indices=torch.from_numpy(indices).to(device),
        spatial_shape=spatial_shape,
        batch_size=1,
    )
    print(f"加载文件: {file_path}, 有效点数: {valid.sum()}, 稀疏位置: {len(unique_linear)}")
    return sparse_tensor


def prepare_sparse_input_from_pcd(
    pcd_path: PathLike,
    spatial_shape: List[int] = [512, 128],
    device: torch.device = torch.device("cuda"),
    add_offset: bool = True,
) -> spconv.SparseConvTensor:
    return load_single_pcd_file(
        pcd_path, spatial_shape, c_slices=64, device=device, add_offset=add_offset
    )


def prepare_dense_input_from_sparse(sparse_tensor: spconv.SparseConvTensor) -> torch.Tensor:
    return sparse_tensor.dense()


def save_points_to_pcd(points: np.ndarray, output_path: PathLike) -> None:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points 形状应为 [N,3+]，实际为 {points.shape}")
    xyz = np.ascontiguousarray(points[:, :3])
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS RangeBin AziBin EleBin
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(xyz)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(xyz)}
DATA binary
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(header.encode("utf-8"))
        handle.write(xyz.tobytes())


def run_batch_inference(
    model: nn.Module,
    input_dir: PathLike,
    device: torch.device,
    model_type: str = "zynq",
    is_super_resolution: bool = True,
    use_amp: bool = True,
    amp_dtype: str = "fp16",
    threshold: float = 0.5,
    batch_size: int = 1,
    output_dir: Optional[PathLike] = None,
    save_output: bool = False,
    save_format: str = "npy",
    add_offset: bool = True,
) -> Dict[str, np.ndarray]:
    """Run directory inference sequentially without the unavailable dataset module."""
    del is_super_resolution, batch_size  # retained for source compatibility
    if save_format not in {"npy", "pcd"}:
        raise ValueError("save_format 必须为 'npy' 或 'pcd'")

    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.pcd"))
    results: Dict[str, np.ndarray] = {}
    for index, path in enumerate(files, start=1):
        sparse = load_single_pcd_file(path, device=device, add_offset=add_offset)
        model_input: InferenceInput = sparse.dense() if model_type == "orin" else sparse
        points = run_inference(
            model,
            model_input,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            threshold=threshold,
            model_type=model_type,
        )
        if isinstance(points, list):
            points = points[0]
        points_np = points.detach().cpu().numpy()
        results[path.name] = points_np

        if save_output and output_dir is not None:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            if save_format == "npy":
                np.save(destination / f"{path.stem}_points.npy", points_np)
            else:
                save_points_to_pcd(points_np, destination / f"{path.stem}_points.pcd")
        if index % 10 == 0:
            print(f"已处理 {index}/{len(files)} 个文件")
    return results


class RadarSRInference:
    """Unified wrapper for single-input and directory inference."""

    def __init__(
        self,
        checkpoint_path: PathLike,
        model_type: str = "orin",
        is_super_resolution: bool = True,
        use_amp: bool = True,
        amp_dtype: str = "fp16",
        device: Optional[torch.device] = None,
        threshold: float = 0.01,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = load_model(
            checkpoint_path,
            self.device,
            model_type=model_type,
            is_super_resolution=is_super_resolution,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        self.model_type = model_type.lower()
        self.is_super_resolution = is_super_resolution
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.threshold = threshold
        print(f"RadarSRInference 已初始化，设备: {self.device}")

    def infer_single(
        self,
        input_tensor: InferenceInput,
        threshold: Optional[float] = None,
        return_probability: bool = False,
    ) -> PointResult:
        return run_inference(
            self.model,
            input_tensor,
            self.device,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            threshold=self.threshold if threshold is None else threshold,
            model_type=self.model_type,
            return_probability=return_probability,
        )

    def infer_batch(
        self, input_tensor: InferenceInput, threshold: Optional[float] = None
    ) -> PointResult:
        return self.infer_single(input_tensor, threshold=threshold)

    def infer_directory(
        self,
        input_dir: PathLike,
        output_dir: Optional[PathLike] = None,
        batch_size: int = 1,
        threshold: Optional[float] = None,
        save_output: bool = False,
        save_format: str = "npy",
        add_offset: bool = True,
    ) -> Dict[str, np.ndarray]:
        return run_batch_inference(
            self.model,
            input_dir,
            self.device,
            model_type=self.model_type,
            is_super_resolution=self.is_super_resolution,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            threshold=self.threshold if threshold is None else threshold,
            batch_size=batch_size,
            output_dir=output_dir,
            save_output=save_output,
            save_format=save_format,
            add_offset=add_offset,
        )

    def get_output_shape(self) -> Tuple[int, int, int]:
        return (64, 2048, 512) if self.is_super_resolution else (32, 1024, 256)

    @staticmethod
    def convert_to_dense(sparse_tensor: spconv.SparseConvTensor) -> torch.Tensor:
        return sparse_tensor.dense()


def get_inference_interface():
    return {
        "load_model": load_model,
        "run_inference": run_inference,
        "extract_points_from_output": extract_points_from_output,
        "load_single_pcd_file": load_single_pcd_file,
        "save_points_to_pcd": save_points_to_pcd,
        "RadarSRInference": RadarSRInference,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="雷达超分辨率模型推理")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--input_file")
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir", default="./output/infer")
    parser.add_argument("--model_type", choices=("zynq", "orin"), default="zynq")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--is_super_resolution", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_format", choices=("npy", "pcd"), default="npy")
    parser.add_argument(
        "--add_offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="对 RCS 和补偿 Doppler 加 70（训练格式要求，默认开启）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    inferencer = RadarSRInference(
        args.checkpoint_path,
        model_type=args.model_type,
        is_super_resolution=args.is_super_resolution,
        use_amp=args.use_amp,
        device=device,
        threshold=args.threshold,
    )

    if args.input_file:
        sparse = load_single_pcd_file(args.input_file, device=device, add_offset=args.add_offset)
        model_input: InferenceInput = sparse.dense() if args.model_type == "orin" else sparse
        points = inferencer.infer_single(model_input)
        if isinstance(points, list):
            points = points[0]
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        points_np = points.detach().cpu().numpy()
        if args.save_format == "npy":
            np.save(output_dir / f"{Path(args.input_file).stem}_points.npy", points_np)
        else:
            save_points_to_pcd(points_np, output_dir / f"{Path(args.input_file).stem}_points.pcd")
        print(f"提取点数: {len(points_np)}")
    elif args.input_dir:
        inferencer.infer_directory(
            args.input_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            save_output=True,
            save_format=args.save_format,
            add_offset=args.add_offset,
        )
    else:
        print("未指定输入，请使用 --input_file 或 --input_dir")


__all__ = [
    "load_model",
    "load_quantized_model",
    "run_inference",
    "run_batch_inference",
    "extract_points_from_output",
    "save_points_to_pcd",
    "load_single_pcd_file",
    "RadarSRInference",
    "prepare_dense_input_from_sparse",
    "prepare_sparse_input_from_pcd",
    "get_inference_interface",
]


if __name__ == "__main__":
    main()
