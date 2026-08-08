# 原脚本第 1-6 行仅为注释与说明，无可执行代码。





import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import math
import spconv.pytorch as spconv
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
import json  # 新增：用于保存评估报告
import pandas as pd  # AAI处理需要

# Numba JIT 加速
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 如果 numba 不可用，使用普通函数
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

# 可视化相关
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# 导入模型模块 - 使用 run_inference 提供的推理接口
from run_inference import load_model, run_inference, RadarSRInference, extract_points_from_output


# ==================== Occ真值评估模块 ====================
GRID_RES = 0.6  # 网格分辨率（米），必须与Occ生成时一致

# 坐标转换系数（与dataset_inference.py保持一致）
RANGE_COEF_METADATA = 40960
AZIMUTH_COEF_METADATA = 1047
ELEVATION_COEF_METADATA = 1397
RANGE_ZOOM = 3

# AAI处理列索引常量
COL_RANGE = 0
COL_AZI = 1
COL_ELE = 2
COL_DOPPLER = 3
COL_RCS = 4
COL_PHASE = 5

# AAI处理列名（与 aai_test.py 保持一致）
COL_NAMES = ['RangeBin', 'AziBin', 'EleBin', 'AbsDoppler', 'RCS', 'PhaseBin']

# Bin 维度
RANGE_DIM = 512
AZI_DIM = 128
ELE_DIM = 32

# 动静态划分阈值（备用固定阈值）
DOPPLER_DYNAMIC_THRESHOLD = 1.5

# 动态点云AAI处理参数（仅作用于动态点云三阶段NMS；静态固定 thresh=1、keep_neighbors=False）
DYNAMIC_CONTINUITY_THRESHOLD = 1  # 动态点云连续性判断阈值（差值 <= 该值视为同组）
DYNAMIC_NEIGHBOR_THRESHOLD = 3    # 动态点云邻居判定差值阈值（|target-max_target| <= 该值则保留）

# 雷达坐标参数
RADAR_X = 0.0  # 雷达X坐标
RADAR_Y = 3.7  # 雷达Y坐标

# FFT范围参数
AZI_FFT_RANGE = 128
ELE_FFT_RANGE = 32

# VThrsi计算的误差参数
DELTA_VX = 0.1             # Vx误差
DELTA_VY = 0.1             # Vy误差
DELTA_SIN_AZI = 0.01745    # SinAzi误差（对应1°）
DELTA_SIN_ELE = 0.03490    # SinEle误差（对应2°）
DELTA_AZI = 0.01745        # 方位角误差（1° in rad）
DELTA_ELE = 0.01745        # 俯仰角误差（1° in rad）
DELTA_DOP = 0.1            # doppler误差

def xyz_to_grid_index(xyz: np.ndarray, grid_res: float = GRID_RES) -> np.ndarray:
    """
    将物理坐标(x,y,z)转换为Occ网格索引 (GridY, GridX, GridZ)
    与SuperPoint_TruthValue_main.py兼容

    转换逻辑:
    - GridY: 从1开始，y>=0时 GridY = int(y/grid_res) + 1
    - GridX: 有正负，正值从1开始，负值从-1开始
    - GridZ: 同GridX逻辑
    """
    if xyz.shape[0] == 0:
        return np.empty((0, 3), dtype=int)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # Y轴：从1开始
    grid_y = (y / grid_res).astype(int) + 1

    # X轴：有正负
    grid_x = (x / grid_res).astype(int)
    grid_x[x >= 0] += 1
    grid_x[x < 0] -= 1

    # Z轴：同X轴逻辑
    grid_z = (z / grid_res).astype(int)
    grid_z[z >= 0] += 1
    grid_z[z < 0] -= 1

    return np.stack([grid_y, grid_x, grid_z], axis=1)


def load_occ_gt_coord(occ_path: str) -> np.ndarray:
    """
    加载坐标格式的Occ真值文件
    返回: Nx4数组 (GridY, GridX, GridZ, Label)
    """
    if not os.path.exists(occ_path):
        return None
    data = np.fromfile(occ_path, dtype=np.float32)
    if len(data) == 0:
        return np.empty((0, 4), dtype=np.float32)
    return data.reshape(-1, 4)


def evaluate_frame(xyz_points: np.ndarray, occ_data: np.ndarray,
                   grid_res: float = GRID_RES) -> Dict:
    """
    计算单帧点云的 ACC, IoU, PPV

    参数:
        xyz_points: 点云物理坐标 [N, 3]
        occ_data: Occ真值 [M, 4] (GridY, GridX, GridZ, Label)
        grid_res: 网格分辨率

    返回:
        评估结果字典
    """
    if xyz_points.shape[0] == 0:
        return {
            'acc': 0.0, 'iou': 0.0, 'ppv': 0.0,
            'tp': 0, 'total_points': 0, 'hit_voxels': 0, 'total_occ_voxels': 0
        }

    if occ_data is None or occ_data.shape[0] == 0:
        return {
            'acc': 0.0, 'iou': 0.0, 'ppv': 0.0,
            'tp': 0, 'total_points': len(xyz_points), 'hit_voxels': 0, 'total_occ_voxels': 0
        }

    # 提取有效Occ网格（Label=2或3）
    valid_mask = (occ_data[:, 3] == 2) | (occ_data[:, 3] == 3)
    valid_occ = occ_data[valid_mask]
    total_valid_occ = valid_occ.shape[0]

    if total_valid_occ == 0:
        return {
            'acc': 0.0, 'iou': 0.0, 'ppv': 0.0,
            'tp': 0, 'total_points': len(xyz_points), 'hit_voxels': 0, 'total_occ_voxels': 0
        }

    # 构造查找集合
    valid_occ_set = set(map(tuple, valid_occ[:, :3].astype(int)))

    # 点云转网格索引
    point_grid_indices = xyz_to_grid_index(xyz_points, grid_res)

    # 统计命中情况
    hit_voxels = set()
    tp = 0
    for idx in point_grid_indices:
        idx_tuple = tuple(idx)
        if idx_tuple in valid_occ_set:
            tp += 1
            hit_voxels.add(idx_tuple)

    total_points = len(xyz_points)
    acc = tp / total_points if total_points > 0 else 0.0
    iou = len(hit_voxels) / total_valid_occ if total_valid_occ > 0 else 0.0
    ppv = tp / len(hit_voxels) if len(hit_voxels) > 0 else 0.0

    return {
        'acc': acc, 'iou': iou, 'ppv': ppv,
        'tp': tp, 'total_points': total_points,
        'hit_voxels': len(hit_voxels), 'total_occ_voxels': total_valid_occ
    }


# ==================== AAI精度提升核心函数（与 aai_test.py 一致，pandas 实现） ====================
def calculate_xyz(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据 RangeBin, AziBin, EleBin 计算 xyz 物理坐标

    坐标系变换逻辑:
    - 原始坐标系: x_ori = range * sin(alpha), z = range * sin(ele)
    - y_ori = sqrt(range^2 - x_ori^2 - z^2)
    - 变换后: x = y_ori（纵向变新x）, y = x_ori * (-1)（横向取负变新y）

    Args:
        df: DataFrame包含 RangeBin, AziBin, EleBin 列

    Returns:
        (x, y, z) 坐标数组
    """
    range_bins = df['RangeBin'].astype(np.float32).values
    azi_bins = df['AziBin'].astype(np.float32).values
    ele_bins = df['EleBin'].astype(np.float32).values

    range_coef = RANGE_COEF_METADATA / 65536.0
    azimuth_coef = AZIMUTH_COEF_METADATA / 65536.0
    elevation_coef = ELEVATION_COEF_METADATA / 65536.0

    azi_half = AZI_DIM // 2  # 64
    ele_half = ELE_DIM // 2  # 16

    r = range_coef * (range_bins - 0.5 * RANGE_ZOOM)
    sin_alpha = (azi_bins * (-1) + azi_half) * azimuth_coef
    sin_ele = (ele_bins * (-1) + ele_half) * elevation_coef

    x_ori = r * sin_alpha
    z = r * sin_ele

    y_squared = r**2 - x_ori**2 - z**2
    y_squared = np.maximum(y_squared, 0)
    y_ori = np.sqrt(y_squared)

    x = y_ori          # SR_x: 纵向变成新的x
    y = x_ori * (-1)   # SR_y: 横向取负变成新的y

    return x, y, z


def calculate_vthrs(ele_bins: np.ndarray, azi_bins: np.ndarray,
                    vx: float, vy: float,
                    azimuth_coef: float, elevation_coef: float,
                    padding: float = 0.0) -> np.ndarray:
    """
    计算动态阈值 VThrs

    VThrs = VThrsVy + VThrsVx + deltaDop

    Args:
        ele_bins: EleBin数组
        azi_bins: AziBin数组
        vx: RadarVx (YawRate * RadarY)
        vy: RadarVy (VehicleSpeed - YawRate * RadarX)
        azimuth_coef: 方位角分辨率系数
        elevation_coef: 俯仰角分辨率系数
        padding: padding参数（默认0）

    Returns:
        VThrs数组
    """
    sin_ele = -1.0 * (ele_bins - ELE_FFT_RANGE / 2 + padding) * elevation_coef
    sin_alpha = -1.0 * (azi_bins - AZI_FFT_RANGE / 2 + padding) * azimuth_coef
    cos_ele = np.sqrt(np.maximum(1.0 - sin_ele * sin_ele, 0.0))
    sin_azi = sin_alpha / cos_ele

    src_cos_ele = np.sqrt(np.maximum(1.0 - sin_ele * sin_ele, 0.0))
    src_cos_azi = np.sqrt(np.maximum(1.0 - sin_azi * sin_azi, 0.0))

    vthrs_vy = np.abs(DELTA_VY * src_cos_azi * src_cos_ele) \
             + np.abs(vy * src_cos_azi * DELTA_SIN_ELE) \
             + np.abs(vy * src_cos_azi * sin_ele * DELTA_ELE) \
             + np.abs(vy * DELTA_SIN_AZI * src_cos_ele) \
             + np.abs(vy * DELTA_AZI * sin_azi * src_cos_ele)

    vthrs_vx = np.abs(DELTA_VX * sin_azi * src_cos_ele) \
             + np.abs(vx * sin_azi * DELTA_SIN_ELE) \
             + np.abs(vx * sin_azi * sin_ele * DELTA_ELE) \
             + np.abs(vx * DELTA_SIN_AZI * src_cos_ele) \
             + np.abs(vx * DELTA_AZI * src_cos_azi * src_cos_ele)

    vthrs = vthrs_vy + vthrs_vx + DELTA_DOP

    return vthrs


def nms_filter_pandas(df, sort_cols, group_cols, target_continuity_col,
                      continuity_threshold=1, keep_neighbors=False,
                      neighbor_threshold=1):
    """
    模拟 MATLAB 代码中的 while 循环过滤逻辑。

    逻辑：在 group_cols 相同的情况下，如果 target_continuity_col 是连续的，
    则视为同一组，并只保留组内 RCS 最大的一行。

    Args:
        df: 输入 DataFrame
        sort_cols: 排序列列表
        group_cols: 分组列（固定不变）
        target_continuity_col: 检查连续性的列
        continuity_threshold: 连续性判断阈值，差值 <= threshold 视为连续（默认1）
        keep_neighbors: 是否保留RCS最大点的所有邻居（默认False）
            True时保留每段最大点 + 所有满足 |target-max_target| <= neighbor_threshold 的点
        neighbor_threshold: 邻居判定的差值阈值（默认1）

    Returns:
        过滤后的 DataFrame
    """
    if df.empty:
        return df

    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    condition = pd.Series(False, index=df.index)
    for col in group_cols:
        condition |= (df[col] != df[col].shift(1))
    condition |= (np.abs(df[target_continuity_col] - df[target_continuity_col].shift(1)) > continuity_threshold)
    group_ids = condition.cumsum()

    # 纯 numpy 分段 argmax（比 pandas groupby+idxmax 快 ~100x）：
    # threshold=1 时分组数=N，pandas 逐组 idxmax 开销极大；reduceat 为 O(N)
    vals = df['RCS'].to_numpy()
    target_vals = df[target_continuity_col].to_numpy()
    g = group_ids.to_numpy()
    n = len(vals)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(g) != 0) + 1))
    seg_len = np.diff(np.concatenate([starts, [n]]))
    # 每段 RCS 最大值
    seg_max = np.maximum.reduceat(vals, starts)
    row_max = np.repeat(seg_max, seg_len)
    is_max_row = vals == row_max
    # 每段最大值首次出现位置（等价 idxmax）
    pos = np.where(is_max_row, np.arange(n), n)
    max_positions = np.minimum.reduceat(pos, starts)

    if not keep_neighbors:
        return df.iloc[max_positions].reset_index(drop=True)

    # keep_neighbors: 保留每段RCS最大点 + 所有满足 |target-max_target| <= neighbor_threshold 的点
    # 注意：最大点位置差值为0，已被is_neighbor包含，无需单独添加
    max_target_per_seg = target_vals[max_positions]
    row_max_target = np.repeat(max_target_per_seg, seg_len)
    is_neighbor = np.abs(target_vals - row_max_target) <= neighbor_threshold
    keep_indices = np.where(is_neighbor)[0]
    return df.iloc[keep_indices].reset_index(drop=True)


def aai_frame3(raw_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    对原始雷达点云进行三阶段角精度提升处理

    处理流程:
    1. 第一阶段: Range, Ele 相同, Azi 连续 -> 取 RCS 最大
    2. 第二阶段: Range, Azi 相同, Ele 连续 -> 取 RCS 最大
    3. 第三阶段: Range 相同, Azi 连续 -> 取 RCS 最大

    Args:
        raw_data: 原始雷达数据 [N, 6] (RangeBin, AziBin, EleBin, AbsDoppler, RCS, PhaseBin)

    Returns:
        (x_aai, y_aai, z_aai, df_final): AAI处理后的坐标和DataFrame
    """
    df_ori = pd.DataFrame(raw_data, columns=COL_NAMES)

    # === Stage 1: Range, Ele same, Azi continuous ===
    df_stage1 = nms_filter_pandas(
        df_ori,
        sort_cols=['RangeBin', 'EleBin', 'AziBin'],
        group_cols=['RangeBin', 'EleBin'],
        target_continuity_col='AziBin'
    )
    print(f"  Stage1 (Azi cont): {len(df_ori)} -> {len(df_stage1)} pts")

    # === Stage 2: Range, Azi same, Ele continuous ===
    df_stage2 = nms_filter_pandas(
        df_stage1,
        sort_cols=['RangeBin', 'AziBin', 'EleBin'],
        group_cols=['RangeBin', 'AziBin'],
        target_continuity_col='EleBin'
    )
    print(f"  Stage2 (Ele cont): {len(df_stage1)} -> {len(df_stage2)} pts")

    # === Stage 3: Range same, Azi continuous ===
    df_final = nms_filter_pandas(
        df_stage2,
        sort_cols=['RangeBin', 'AziBin', 'EleBin'],
        group_cols=['RangeBin'],
        target_continuity_col='AziBin'
    )
    print(f"  Stage3 (Range+Azi): {len(df_stage2)} -> {len(df_final)} pts")

    x_aai, y_aai, z_aai = calculate_xyz(df_final)

    return x_aai, y_aai, z_aai, df_final


def aai_frame3_dynamic_static(raw_data: np.ndarray,
                              metadata: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    对原始雷达点云进行动静态分离的三阶段角精度提升处理

    动态点云判断：AbsDoppler >= VThrs（动态阈值）
    动态点云处理：
        - 连续性阈值放宽到 3（差值 <= 3 视为连续）
        - 组内保留 RCS 最大点及其相邻点中 RCS 次大的点

    静态点云处理：
        - 保持原有处理逻辑（连续性阈值 1，组内保留 RCS 最大点）

    Args:
        raw_data: 原始雷达数据 [N, 6] (RangeBin, AziBin, EleBin, AbsDoppler, RCS, PhaseBin)
        metadata: 元数据字典，需包含 VehicleSpeed 和 YawRate

    Returns:
        (x_aai, y_aai, z_aai, df_final): AAI处理后的坐标和DataFrame
    """
    df_ori = pd.DataFrame(raw_data, columns=COL_NAMES)

    # === 动静态分离（使用VThrs和VThrs2两级阈值） ===
    vehicle_speed = metadata.get('VehicleSpeed', 0.0) if metadata else 0.0
    yaw_rate = metadata.get('YawRate', 0.0) if metadata else 0.0

    radar_vx = yaw_rate * RADAR_Y
    radar_vy = vehicle_speed - yaw_rate * RADAR_X  # = vehicle_speed (因为 RadarX=0)

    azimuth_coef = AZIMUTH_COEF_METADATA / 65536.0
    elevation_coef = ELEVATION_COEF_METADATA / 65536.0
    range_coef = RANGE_COEF_METADATA / 65536.0

    # === 第一步：计算VThrs（第一级阈值） ===
    vthrs = calculate_vthrs(
        ele_bins=df_ori['EleBin'].values,
        azi_bins=df_ori['AziBin'].values,
        vx=radar_vx,
        vy=radar_vy,
        azimuth_coef=azimuth_coef,
        elevation_coef=elevation_coef,
        padding=0.0
    )

    # === 第二步：计算物理坐标和range ===
    ranges = range_coef * (df_ori['RangeBin'].values - 0.5 * RANGE_ZOOM)

    azi_half = AZI_DIM // 2  # 64
    ele_half = ELE_DIM // 2  # 16
    sin_alpha = (df_ori['AziBin'].values * (-1) + azi_half) * azimuth_coef
    sin_ele = (df_ori['EleBin'].values * (-1) + ele_half) * elevation_coef

    point_x = ranges * sin_alpha
    point_z = ranges * sin_ele
    point_y_squared = ranges**2 - point_x**2 - point_z**2
    point_y_squared = np.maximum(point_y_squared, 0.0)
    point_y = np.sqrt(point_y_squared)

    # === 第三步：计算SpeedPred和VThrs2（第二级阈值） ===
    speed_pred = (radar_vx * point_x + radar_vy * point_y) / ranges
    vthrs2 = speed_pred - radar_vy - 0.1

    # === 第四步：计算SpeedError并判断动静态 ===
    # SpeedError = doppler + SpeedPred = 目标绝对速度
    # 旧格式(6字段)：pcd中的AbsDoppler已经是SpeedError
    # 新格式(18字段)：doppler列为动态参考的(含ego-motion)，需要计算 SpeedError = doppler + SpeedPred
    doppler_is_raw = metadata.get('doppler_is_raw', False) if metadata else False
    if doppler_is_raw:
        speed_error = df_ori['AbsDoppler'].values + speed_pred
        print(f"  [New format] Computed SpeedError from raw doppler + SpeedPred")
    else:
        speed_error = df_ori['AbsDoppler'].values

    # === 第五步：两级阈值判断 ===
    point_validity = np.zeros(len(df_ori), dtype=np.int8)

    abs_speed_error = np.abs(speed_error)

    cond1 = abs_speed_error > vthrs
    cond2 = (speed_error < -vthrs) & (speed_error > vthrs2)

    point_validity[cond1 & cond2] = 2   # 其他类型
    point_validity[cond1 & ~cond2] = 8  # 动态点云
    point_validity[~cond1] = 9          # 静态点云

    dynamic_mask = point_validity == 8
    df_dynamic = df_ori[dynamic_mask].reset_index(drop=True)
    df_static = df_ori[~dynamic_mask].reset_index(drop=True)

    df_ori['PointValidity'] = point_validity

    n_dynamic = np.sum(point_validity == 8)
    n_static = np.sum(point_validity == 9)
    n_other = np.sum(point_validity == 2)

    print(f"  VThrs  stats: min={np.min(vthrs):.4f}, max={np.max(vthrs):.4f}, mean={np.mean(vthrs):.4f}")
    print(f"  VThrs2 stats: min={np.min(vthrs2):.4f}, max={np.max(vthrs2):.4f}, mean={np.mean(vthrs2):.4f}")
    print(f"  SpeedError stats: min={np.min(speed_error):.4f}, max={np.max(speed_error):.4f}, mean={np.mean(speed_error):.4f}")
    print(f"  Point validity: Dynamic(8)={n_dynamic}, Static(9)={n_static}, Other(2)={n_other}")
    print(f"  Dynamic/Static split (两级阈值): Original {len(df_ori)} pts -> Dynamic {len(df_dynamic)} pts, Static {len(df_static)} pts")
    print(f"  VehicleSpeed={vehicle_speed:.2f}, YawRate={yaw_rate:.4f}, RadarVx={radar_vx:.4f}, RadarVy={radar_vy:.2f}")

    # === 动态点云处理（宽松阈值 + 保留相邻点） ===
    if len(df_dynamic) > 0:
        df_dyn_s1 = nms_filter_pandas(
            df_dynamic,
            sort_cols=['RangeBin', 'EleBin', 'AziBin'],
            group_cols=['RangeBin', 'EleBin'],
            target_continuity_col='AziBin',
            continuity_threshold=DYNAMIC_CONTINUITY_THRESHOLD,
            keep_neighbors=True,
            neighbor_threshold=DYNAMIC_NEIGHBOR_THRESHOLD
        )
        print(f"  Dynamic Stage1 (Azi cont, cthresh={DYNAMIC_CONTINUITY_THRESHOLD}, nb={DYNAMIC_NEIGHBOR_THRESHOLD}): {len(df_dynamic)} -> {len(df_dyn_s1)} pts")

        df_dyn_s2 = nms_filter_pandas(
            df_dyn_s1,
            sort_cols=['RangeBin', 'AziBin', 'EleBin'],
            group_cols=['RangeBin', 'AziBin'],
            target_continuity_col='EleBin',
            continuity_threshold=DYNAMIC_CONTINUITY_THRESHOLD,
            keep_neighbors=True,
            neighbor_threshold=DYNAMIC_NEIGHBOR_THRESHOLD
        )
        print(f"  Dynamic Stage2 (Ele cont, cthresh={DYNAMIC_CONTINUITY_THRESHOLD}, nb={DYNAMIC_NEIGHBOR_THRESHOLD}): {len(df_dyn_s1)} -> {len(df_dyn_s2)} pts")

        df_dyn_final = nms_filter_pandas(
            df_dyn_s2,
            sort_cols=['RangeBin', 'AziBin', 'EleBin'],
            group_cols=['RangeBin'],
            target_continuity_col='AziBin',
            continuity_threshold=DYNAMIC_CONTINUITY_THRESHOLD,
            keep_neighbors=True,
            neighbor_threshold=DYNAMIC_NEIGHBOR_THRESHOLD
        )
        print(f"  Dynamic Stage3 (Range+Azi, cthresh={DYNAMIC_CONTINUITY_THRESHOLD}, nb={DYNAMIC_NEIGHBOR_THRESHOLD}): {len(df_dyn_s2)} -> {len(df_dyn_final)} pts")
        df_dyn_final['IsDynamic'] = True
    else:
        df_dyn_final = pd.DataFrame(columns=COL_NAMES + ['IsDynamic'])

    # === Static point processing (original logic) ===
    if len(df_static) > 0:
        df_sta_s1 = nms_filter_pandas(
            df_static,
            sort_cols=['RangeBin', 'EleBin', 'AziBin'],
            group_cols=['RangeBin', 'EleBin'],
            target_continuity_col='AziBin',
            continuity_threshold=1,
            keep_neighbors=False
        )
        print(f"  Static Stage1 (Azi cont, thresh=1): {len(df_static)} -> {len(df_sta_s1)} pts")

        df_sta_s2 = nms_filter_pandas(
            df_sta_s1,
            sort_cols=['RangeBin', 'AziBin', 'EleBin'],
            group_cols=['RangeBin', 'AziBin'],
            target_continuity_col='EleBin',
            continuity_threshold=1,
            keep_neighbors=False
        )
        print(f"  Static Stage2 (Ele cont, thresh=1): {len(df_sta_s1)} -> {len(df_sta_s2)} pts")

        df_sta_final = nms_filter_pandas(
            df_sta_s2,
            sort_cols=['RangeBin', 'AziBin', 'EleBin'],
            group_cols=['RangeBin'],
            target_continuity_col='AziBin',
            continuity_threshold=1,
            keep_neighbors=False
        )
        print(f"  Static Stage3 (Range+Azi, thresh=1): {len(df_sta_s2)} -> {len(df_sta_final)} pts")
        df_sta_final['IsDynamic'] = False
    else:
        df_sta_final = pd.DataFrame(columns=COL_NAMES + ['IsDynamic'])

    # === Merge dynamic and static results ===
    df_final = pd.concat([df_dyn_final, df_sta_final], ignore_index=True)
    print(f"  Merged: Dynamic {len(df_dyn_final)} + Static {len(df_sta_final)} = {len(df_final)} pts")

    x_aai, y_aai, z_aai = calculate_xyz(df_final)

    return x_aai, y_aai, z_aai, df_final, point_validity


def merge_sr_with_aai_bins(sr_points: np.ndarray, aai_data: np.ndarray,
                           sr_rate1: int = 4, sr_rate2: int = 2) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    将超分模型输出点云与AAI处理后的点云在bin坐标层面叠加合并

    Args:
        sr_points: 超分点云 [N, 3] (RangeBin, AziBin, EleBin) - 已是SR目标分辨率
        aai_data: AAI处理后的bin数据 [M, 6] (RangeBin, AziBin, EleBin, AbsDoppler, RCS, PhaseBin)
                  - 原始雷达分辨率，需要缩放到SR分辨率
        sr_rate1: Range/Azi超分倍数（默认4）
        sr_rate2: Ele超分倍数（默认2）

    Returns:
        (merged_points, merge_stats, is_sr_arr): 合并后的bin坐标点云 [K, 3]、统计信息
        和 is_sr 标记数组 [K]（1=模型SR生成，0=AAI叠加的原始点云）。
        重叠点视为SR生成(is_sr=1)。
    """
    # 提取bin坐标集合
    sr_bin_set = set()
    if sr_points.shape[0] > 0:
        for i in range(sr_points.shape[0]):
            r_bin = int(round(sr_points[i, 0]))
            a_bin = int(round(sr_points[i, 1]))
            e_bin = int(round(sr_points[i, 2]))
            sr_bin_set.add((r_bin, a_bin, e_bin))

    # AAI点云的bin坐标 - 需要缩放到SR分辨率
    aai_bin_set = set()
    if aai_data.shape[0] > 0:
        for i in range(aai_data.shape[0]):
            r_bin_raw = aai_data[i, COL_RANGE]
            a_bin_raw = aai_data[i, COL_AZI]
            e_bin_raw = aai_data[i, COL_ELE]
            # 缩放到SR分辨率
            r_bin = int(round(r_bin_raw * sr_rate1))
            a_bin = int(round(a_bin_raw * sr_rate1))
            e_bin = int(round(e_bin_raw * sr_rate2))
            aai_bin_set.add((r_bin, a_bin, e_bin))

    # 合并bin集合并追踪每个点来源
    # is_sr = 1: 点在SR集中（模型生成，覆盖SR独有和重叠）
    # is_sr = 0: 点仅在AAI集中（原始雷达，经AAI叠加上去的）
    merged_list = []
    is_sr_list = []
    # 先加入所有SR点 (is_sr = 1)
    for bin_tuple in sr_bin_set:
        merged_list.append(bin_tuple)
        is_sr_list.append(1)
    # 再加入AAI独有的点 (is_sr = 0)
    aai_only_set = aai_bin_set - sr_bin_set
    for bin_tuple in aai_only_set:
        merged_list.append(bin_tuple)
        is_sr_list.append(0)

    # 生成合并后的点云bin坐标数组
    if len(merged_list) > 0:
        merged_points = np.array(merged_list, dtype=np.float32)
        is_sr_arr = np.array(is_sr_list, dtype=np.uint8)
    else:
        merged_points = np.empty((0, 3), dtype=np.float32)
        is_sr_arr = np.empty((0,), dtype=np.uint8)

    # 统计信息
    merge_stats = {
        'sr_points': len(sr_points),
        'sr_unique_bins': len(sr_bin_set),
        'aai_points': len(aai_data),
        'aai_unique_bins': len(aai_bin_set),
        'merged_bins': len(merged_list),
        'overlap_bins': len(sr_bin_set & aai_bin_set),
        'sr_only_bins': len(sr_bin_set - aai_bin_set),
        'aai_only_bins': len(aai_only_set)
    }

    return merged_points, merge_stats, is_sr_arr


# ==================== PCD文件解析 ====================
def parse_pcd_header(file_path: str) -> Tuple[Dict, List[str], int, List[str]]:
    """
    解析PCD文件头，提取元数据信息

    返回:
        metadata: 元数据字典（包含RadarTimestamp, VehicleSpeed等）
        fields: 字段名列表
        num_points: 点云数量
        header_lines: 头部行列表
    """
    metadata = {}
    fields = []
    num_points = 0
    header_lines = []

    with open(file_path, 'rb') as f:
        while True:
            line = f.readline()
            try:
                line_str = line.decode('utf-8').strip()
            except:
                break

            header_lines.append(line_str)

            # 解析元数据注释行（支持 '# metadata:' 和 '# metadata' 两种前缀）
            if line_str.startswith('# metadata:') or line_str.startswith('# metadata'):
                if line_str.startswith('# metadata:'):
                    metadata_str = line_str.replace('# metadata:', '').strip()
                else:
                    metadata_str = line_str.replace('# metadata', '').strip()
                # 使用正则解析 Key=Value 或 Key = Value 格式
                import re
                pattern = r'(\w+)\s*=\s*([\d.\-]+)'
                for key, value in re.findall(pattern, metadata_str):
                    try:
                        metadata[key] = float(value)
                    except:
                        metadata[key] = value

            # 解析字段定义
            if line_str.startswith('FIELDS'):
                fields = line_str.split()[1:]

            # 解析点数
            if line_str.startswith('POINTS'):
                num_points = int(line_str.split()[1])

            # 遇到DATA行停止
            if line_str.startswith('DATA'):
                break

    # 根据字段名判断 doppler 是否为原始值
    # 新格式(18字段): 有 'doppler' 字段，为原始多普勒(含ego-motion)
    # 旧格式(6字段): 有 'AbsDoppler' 字段，已经是 SpeedError
    metadata['doppler_is_raw'] = 'doppler' in fields

    return metadata, fields, num_points, header_lines


def read_pcd_binary_data(file_path: str, fields: List[str], num_points: int) -> np.ndarray:
    """
    读取PCD文件的二进制点云数据

    返回:
        numpy结构化数组，包含所有字段数据
    """
    # 根据字段定义数据类型
    dtype_list = []
    for field in fields:
        # 整型字段使用int16
        if field in ['RangeBin', 'DopplerBin', 'AziBin', 'EleBin', 'PowerBin', 'PhaseBin']:
            dtype_list.append((field, np.int16))
        else:
            dtype_list.append((field, np.float32))

    dtype = np.dtype(dtype_list)

    # 读取二进制数据
    with open(file_path, 'rb') as f:
        # 跳过头部
        while True:
            line = f.readline()
            try:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith('DATA'):
                    break
            except:
                break

        # 读取二进制数据
        data = np.fromfile(f, dtype=dtype, count=num_points)

    return data


def parse_pcd_file(file_path: str) -> Tuple[Dict, np.ndarray, List[str]]:
    """
    解析完整的PCD文件

    返回:
        metadata: 元数据字典
        data: 点云数据的numpy结构化数组
        header_lines: 头部行列表
    """
    metadata, fields, num_points, header_lines = parse_pcd_header(file_path)
    data = read_pcd_binary_data(file_path, fields, num_points)
    return metadata, data, header_lines


# ==================== 数据处理函数 ====================
def create_input_matrix(data: np.ndarray, vehicle_speed: float,
                        range_dim: int = 512, azi_dim: int = 128, ele_dim: int = 32) -> np.ndarray:
    """
    创建U-Net模型的输入矩阵（密集格式，兼容 ORIN 模型）

    输出形状: [64, range_dim, azi_dim]
    - 64 通道 = 32 个 EleBin * 2 个特征 (RCS, Doppler)
    - 每个位置的特征: channel[2*e] = RCS, channel[2*e+1] = Doppler

    注意: 此格式与 EnhancedBinDataset 保持一致，用于推理脚本

    参数:
        data: 原始点云数据（结构化数组）
        vehicle_speed: 车辆速度（用于修正doppler）
        range_dim: Range维度大小（默认512）
        azi_dim: Azimuth维度大小（默认128）
        ele_dim: Elevation维度大小（默认32）

    返回:
        输入矩阵 numpy数组 [64, range_dim, azi_dim]
    """
    # 初始化输入矩阵: 64 通道 = ele_dim * 2
    input_matrix = np.zeros((ele_dim * 2, range_dim, azi_dim), dtype=np.float32)

    # 提取各字段数据
    range_bins = data['RangeBin'].astype(np.int32)
    azi_bins = data['AziBin'].astype(np.int32)
    ele_bins = data['EleBin'].astype(np.int32)
    rcs = data['RCS']
    doppler = data['doppler'] + vehicle_speed  # doppler + VehicleSpeed

    # 将bin值裁剪到有效范围
    range_bins = np.clip(range_bins, 0, range_dim - 1)
    azi_bins = np.clip(azi_bins, 0, azi_dim - 1)
    ele_bins = np.clip(ele_bins, 0, ele_dim - 1)

    # 填充矩阵（与 EnhancedBinDataset 格式一致）
    for i in range(len(data)):
        r, a, e = range_bins[i], azi_bins[i], ele_bins[i]
        # 通道 2*e 存储 RCS
        input_matrix[2 * e, r, a] = rcs[i]
        # 通道 2*e+1 存储 Doppler（已加 VehicleSpeed）
        input_matrix[2 * e + 1, r, a] = doppler[i]

    return input_matrix


def create_sparse_input(data: np.ndarray,
                        range_dim: int = 512, azi_dim: int = 128, ele_dim: int = 32,
                        device: torch.device = torch.device('cuda'),
                        add_offset: bool = True,
                        metadata: Optional[Dict] = None) -> spconv.SparseConvTensor:
    """
    创建稀疏输入张量（用于 ZYNQ 模型或转换为密集张量）

    输出: SparseConvTensor
    - spatial_shape: [range_dim, azi_dim] = [512, 128]
    - features: [N_points, 64] = [N_points, ele_dim * 2]

    参数:
        data: 原始点云数据（结构化数组）
        range_dim: Range维度大小
        azi_dim: Azimuth维度大小
        ele_dim: Elevation维度大小
        device: 计算设备
        add_offset: 是否对 RCS 和 Doppler 加70偏置（与 EnhancedBinDataset 训练一致），默认True
        metadata: 元数据字典（含 VehicleSpeed, YawRate），用于对新格式 doppler 做 ego-motion 补偿。
                  训练时 EnhancedBinDataset 使用的是 AbsDoppler (SpeedError, 已补偿)。
                  新格式(18字段)的 doppler 为原始值，需要计算 SpeedError = doppler + SpeedPred 以保持一致。

    返回:
        SparseConvTensor 稀疏张量
    """
    # 提取各字段数据
    range_bins = data['RangeBin'].astype(np.int32)
    azi_bins = data['AziBin'].astype(np.int32)
    ele_bins = data['EleBin'].astype(np.int32)
    rcs = data['RCS']

    # === Doppler 处理：根据数据格式决定是否需要 ego-motion 补偿 ===
    # 训练 (EnhancedBinDataset): 使用 AbsDoppler (SpeedError, 已补偿) + 70
    # 新格式(18字段): doppler 为原始值，需要计算 SpeedError = doppler + SpeedPred
    # 旧格式(6字段): AbsDoppler 已是 SpeedError，直接使用
    if 'doppler' in data.dtype.names:
        raw_doppler = data['doppler'].astype(np.float32)
        if metadata is not None and 'VehicleSpeed' in metadata:
            # 新格式：计算 SpeedPred 并补偿
            vehicle_speed = metadata.get('VehicleSpeed', 0.0)
            yaw_rate = metadata.get('YawRate', 0.0)
            radar_vx = yaw_rate * RADAR_Y
            radar_vy = vehicle_speed - yaw_rate * RADAR_X  # = vehicle_speed

            range_coef = RANGE_COEF_METADATA / 65536.0
            azimuth_coef = AZIMUTH_COEF_METADATA / 65536.0
            elevation_coef = ELEVATION_COEF_METADATA / 65536.0
            azi_half = azi_dim // 2
            ele_half = ele_dim // 2

            ranges = range_coef * (range_bins.astype(np.float32) - 0.5 * RANGE_ZOOM)
            sin_alpha = (azi_bins.astype(np.float32) * (-1) + azi_half) * azimuth_coef
            sin_ele = (ele_bins.astype(np.float32) * (-1) + ele_half) * elevation_coef

            point_x = ranges * sin_alpha
            point_z = ranges * sin_ele
            point_y_sq = np.maximum(ranges**2 - point_x**2 - point_z**2, 0.0)
            point_y = np.sqrt(point_y_sq)

            ranges_safe = np.where(ranges > 0, ranges, 1.0)
            speed_pred = (radar_vx * point_x + radar_vy * point_y) / ranges_safe

            doppler = raw_doppler + speed_pred
        else:
            # 无 metadata，使用原始 doppler（向前兼容）
            doppler = raw_doppler
    elif 'AbsDoppler' in data.dtype.names:
        # 旧格式：AbsDoppler 已经是 SpeedError
        doppler = data['AbsDoppler'].astype(np.float32)
    else:
        raise ValueError("数据中未找到 doppler 或 AbsDoppler 字段")

    # 裁剪到有效范围
    range_bins = np.clip(range_bins, 0, range_dim - 1)
    azi_bins = np.clip(azi_bins, 0, azi_dim - 1)
    ele_bins = np.clip(ele_bins, 0, ele_dim - 1)

    # 创建唯一坐标 (Range, Azi)
    linear_idx = range_bins * azi_dim + azi_bins
    unique_linear, inverse = np.unique(linear_idx, return_inverse=True)

    # 恢复唯一坐标
    unique_range = unique_linear // azi_dim
    unique_azi = unique_linear % azi_dim

    # 为每个唯一位置创建 64 维特征向量
    features = np.zeros((len(unique_linear), ele_dim * 2), dtype=np.float32)

    # 向量化填充特征（替代Python循环）
    # 特征索引: channel = 2*ele 或 2*ele+1
    channel_indices_rcs = 2 * ele_bins
    channel_indices_doppler = 2 * ele_bins + 1

    # 使用 np.add.at 处理同一位置多个点的累加（或直接覆盖）
    # 这里使用直接覆盖，因为后续点会替换前面的点
    if add_offset:
        # 与 EnhancedBinDataset 训练一致: RCS 和 Doppler 加70偏置
        features[inverse, channel_indices_rcs] = rcs + 70
        features[inverse, channel_indices_doppler] = doppler + 70
    else:
        features[inverse, channel_indices_rcs] = rcs
        features[inverse, channel_indices_doppler] = doppler

    # 构建 batch indices [N_points, 3] -> (batch_idx, range, azi)
    indices = np.zeros((len(unique_linear), 3), dtype=np.int32)
    indices[:, 1] = unique_range  # range -> Y (H)
    indices[:, 2] = unique_azi    # azi -> X (W)

    # 转换为 torch tensor
    indices_tensor = torch.from_numpy(indices).to(device)
    features_tensor = torch.from_numpy(features).to(device)

    # 构建 SparseConvTensor
    sparse_tensor = spconv.SparseConvTensor(
        features=features_tensor,
        indices=indices_tensor,
        spatial_shape=[range_dim, azi_dim],
        batch_size=1
    )

    return sparse_tensor


def extract_sr_points_from_output(output: np.ndarray,
                                  sr_rate1: int = 4, sr_rate2: int = 2,
                                  original_data: np.ndarray = None,
                                  threshold: float = 0.01) -> np.ndarray:
    """
    从U-Net输出中提取超分辨率点云

    输出包含: SR_RangeBin, SR_AziBin, SR_EleBin 的预测值

    参数:
        output: 模型输出矩阵 shape (3, D, H, W)
        sr_rate1: Range和Azimuth超分倍数（默认4）
        sr_rate2: Elevation超分倍数（默认2）
        original_data: 原始点云数据（用于引导提取）
        threshold: 阈值（用于判断有效点）

    返回:
        超分辨率点数组 (SR_RangeBin, SR_AziBin, SR_EleBin)
    """
    sr_points = []

    # 输出维度
    sr_range_dim = output.shape[1]
    sr_azi_dim = output.shape[2]
    sr_ele_dim = output.shape[3]

    # 期望的超分维度
    expected_range = 512 * sr_rate1  # 2048
    expected_azi = 128 * sr_rate1    # 512
    expected_ele = 32 * sr_rate2     # 64

    if original_data is not None:
        # 使用原始点云引导提取 - 更高效
        range_bins = original_data['RangeBin'].astype(np.int32)
        azi_bins = original_data['AziBin'].astype(np.int32)
        ele_bins = original_data['EleBin'].astype(np.int32)

        for i in range(len(original_data)):
            # 将原始bin缩放到超分空间
            sr_range_base = range_bins[i] * sr_rate1
            sr_azi_base = azi_bins[i] * sr_rate1
            sr_ele_base = ele_bins[i] * sr_rate2

            # 在输出中查找对应位置
            r_start = max(0, min(sr_range_base, sr_range_dim - 1))
            a_start = max(0, min(sr_azi_base, sr_azi_dim - 1))
            e_start = max(0, min(sr_ele_base, sr_ele_dim - 1))

            # 从输出中提取预测的超分bin值
            sr_range_bin = int(round(output[0, r_start, a_start, e_start] * expected_range))
            sr_azi_bin = int(round(output[1, r_start, a_start, e_start] * expected_azi))
            sr_ele_bin = int(round(output[2, r_start, a_start, e_start] * expected_ele))

            # 验证并添加点
            if 0 <= sr_range_bin < expected_range and \
               0 <= sr_azi_bin < expected_azi and \
               0 <= sr_ele_bin < expected_ele:
                sr_points.append((sr_range_bin, sr_azi_bin, sr_ele_bin))
    else:
        # 遍历整个输出网格
        for r in range(sr_range_dim):
            for a in range(sr_azi_dim):
                for e in range(sr_ele_dim):
                    val0 = output[0, r, a, e]
                    val1 = output[1, r, a, e]
                    val2 = output[2, r, a, e]

                    # 阈值检测
                    if abs(val0) > threshold or abs(val1) > threshold or abs(val2) > threshold:
                        sr_range_bin = r * sr_rate1 if sr_range_dim < expected_range else r
                        sr_azi_bin = a * sr_rate1 if sr_azi_dim < expected_azi else a
                        sr_ele_bin = e * sr_rate2 if sr_ele_dim < expected_ele else e

                        sr_points.append((sr_range_bin, sr_azi_bin, sr_ele_bin))

    return np.array(sr_points) if sr_points else np.empty((0, 3), dtype=np.int32)


# ==================== 坐标计算函数 ====================
def calculate_sr_coordinates_batched(sr_range_bins: np.ndarray, sr_azi_bins: np.ndarray,
                                     sr_ele_bins: np.ndarray,
                                     sr_rate1: int = 4, sr_rate2: int = 2) -> Tuple[np.ndarray, ...]:
    """
    批量计算超分点坐标（向量化版本）

    参数:
        sr_range_bins: 超分RangeBin数组 [N]
        sr_azi_bins: 超分AziBin数组 [N]
        sr_ele_bins: 超分EleBin数组 [N]
        sr_rate1: Range和Azimuth超分倍数
        sr_rate2: Elevation超分倍数

    返回:
        (SR_x, SR_y, SR_z, SR_range, SR_sinAzi, SR_sinEle, valid_mask) 数组
        无效点的坐标为 NaN，valid_mask 标记有效点
    """
    # 系数参数（按用户指定）
    range_coef_metadata = 40960
    azimuth_coef_metadata = 1047
    elevation_coef_metadata = 1397
    range_zoom = 3

    # 计算系数
    range_coef = range_coef_metadata / 65536.0
    azimuth_coef = azimuth_coef_metadata / 65536.0
    elevation_coef = elevation_coef_metadata / 65536.0

    sr_range_coef = range_coef / sr_rate1
    sr_azimuth_coef = azimuth_coef / sr_rate1
    sr_elevation_coef = elevation_coef / sr_rate2

    # 半宽参数
    azi_half = 64
    ele_half = 16
    sr_azi_half = azi_half * sr_rate1
    sr_ele_half = ele_half * sr_rate2

    # 计算range和角度（向量化）
    SR_range = sr_range_coef * (sr_range_bins - 0.5 * range_zoom)
    SR_sinAzi = (sr_azi_bins * (-1) + sr_azi_half) * sr_azimuth_coef
    SR_sinEle = (sr_ele_bins * (-1) + sr_ele_half) * sr_elevation_coef

    # 计算笛卡尔坐标（向量化）
    SR_x_Ori = SR_range * SR_sinAzi
    SR_z = SR_range * SR_sinEle

    # 计算y，检查几何有效性
    y_squared = SR_range**2 - SR_x_Ori**2 - SR_z**2
    valid_mask = y_squared > 0

    # 无效点设为 NaN
    SR_y_Ori = np.sqrt(np.maximum(y_squared, 0))

    # 坐标系变换（按用户指定）
    SR_x = SR_y_Ori
    SR_y = SR_x_Ori * (-1)

    # 无效点坐标设为 NaN
    SR_x = np.where(valid_mask, SR_x, np.nan)
    SR_y = np.where(valid_mask, SR_y, np.nan)
    SR_z = np.where(valid_mask, SR_z, np.nan)
    SR_range = np.where(valid_mask, SR_range, np.nan)
    SR_sinAzi = np.where(valid_mask, SR_sinAzi, np.nan)
    SR_sinEle = np.where(valid_mask, SR_sinEle, np.nan)

    return SR_x, SR_y, SR_z, SR_range, SR_sinAzi, SR_sinEle, valid_mask


def is_valid_fov_batched(sinAzi: np.ndarray, sinEle: np.ndarray) -> np.ndarray:
    """
    批量检查点是否在有效视场角范围内（向量化版本）

    FOV: 水平角 ±50°，俯仰角 ±20°

    参数:
        sinAzi: 方位角sin值数组 [N]
        sinEle: 俯仰角sin值数组 [N]

    返回:
        布尔数组，True表示有效
    """
    azi_threshold = math.sin(math.radians(50))  # ≈ 0.766
    ele_threshold = math.sin(math.radians(20))  # ≈ 0.342

    return (np.abs(sinAzi) <= azi_threshold) & (np.abs(sinEle) <= ele_threshold)


# ==================== 点云可视化函数 ====================
def calculate_original_coordinates(data: np.ndarray,
                                   range_coef_metadata: int = 40960,
                                   azimuth_coef_metadata: int = 1047,
                                   elevation_coef_metadata: int = 1397,
                                   range_zoom: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算原始点云的笛卡尔坐标

    参数:
        data: 原始点云数据（结构化数组）
        range_coef_metadata, azimuth_coef_metadata, elevation_coef_metadata: 系数参数
        range_zoom: range缩放参数

    返回:
        (x, y, z) 坐标数组
    """
    # 提取bin值
    range_bins = data['RangeBin'].astype(np.float32)
    azi_bins = data['AziBin'].astype(np.float32)
    ele_bins = data['EleBin'].astype(np.float32)

    # 系数计算
    range_coef = range_coef_metadata / 65536.0
    azimuth_coef = azimuth_coef_metadata / 65536.0
    elevation_coef = elevation_coef_metadata / 65536.0

    # 半宽参数
    azi_half = 64
    ele_half = 16

    # 计算 range 和角度
    ori_range = range_coef * (range_bins - 0.5 * range_zoom)
    ori_sinAzi = (azi_bins * (-1) + azi_half) * azimuth_coef
    ori_sinEle = (ele_bins * (-1) + ele_half) * elevation_coef

    # 计算笛卡尔坐标
    ori_x_Ori = ori_range * ori_sinAzi
    ori_z = ori_range * ori_sinEle

    # 计算 y
    y_squared = ori_range**2 - ori_x_Ori**2 - ori_z**2
    y_squared = np.maximum(y_squared, 0)  # 防止负值
    ori_y_Ori = np.sqrt(y_squared)

    # 坐标系变换
    ori_x = ori_y_Ori
    ori_y = ori_x_Ori * (-1)

    return ori_x, ori_y, ori_z


def visualize_point_clouds(original_data: np.ndarray,
                           sr_points_np: np.ndarray,
                           valid_points: List[Dict],
                           sr_rate1: int = 4,
                           sr_rate2: int = 2,
                           title: str = "Point Cloud Visualization",
                           save_path: Optional[str] = None):
    """
    可视化点云：第一排 RCS 颜色编码，第二排 Doppler 颜色编码

    参数:
        original_data: 原始点云数据（结构化数组）
        sr_points_np: 模型输出的超分点 [N, 3] (RangeBin, AziBin, EleBin)
        valid_points: 匹配过滤后的有效点列表
        sr_rate1: Range/Azi超分倍数
        sr_rate2: Ele超分倍数
        title: 图像标题
        save_path: 保存路径（可选，不保存则直接显示）
    """
    # 创建 2x3 的子图布局
    fig = plt.figure(figsize=(18, 12))

    # 计算所有点的坐标范围，统一尺度
    ori_x, ori_y, ori_z = calculate_original_coordinates(original_data)

    # 计算模型输出点坐标（如果有）
    sr_x, sr_y, sr_z = None, None, None
    if len(sr_points_np) > 0:
        sr_range_bins = sr_points_np[:, 0].astype(np.float32)
        sr_azi_bins = sr_points_np[:, 1].astype(np.float32)
        sr_ele_bins = sr_points_np[:, 2].astype(np.float32)
        sr_x, sr_y, sr_z, _, _, _, _ = calculate_sr_coordinates_batched(
            sr_range_bins, sr_azi_bins, sr_ele_bins, sr_rate1, sr_rate2
        )
        # 过滤 NaN
        if np.any(np.isnan(sr_x)):
            valid_mask = ~np.isnan(sr_x)
            sr_x, sr_y, sr_z = sr_x[valid_mask], sr_y[valid_mask], sr_z[valid_mask]

    # 计算过滤后点坐标（如果有）
    final_x, final_y, final_z, final_rcs, final_doppler = None, None, None, None, None
    if len(valid_points) > 0:
        final_x = np.array([p['SR_x'] for p in valid_points])
        final_y = np.array([p['SR_y'] for p in valid_points])
        final_z = np.array([p['SR_z'] for p in valid_points])
        final_rcs = np.array([p['RCS'] for p in valid_points])
        final_doppler = np.array([p['doppler'] for p in valid_points])

    # 统一坐标范围：根据所有点云的实际范围
    all_x = np.concatenate([ori_x, sr_x if sr_x is not None else np.array([]), final_x if final_x is not None else np.array([])])
    all_y = np.concatenate([ori_y, sr_y if sr_y is not None else np.array([]), final_y if final_y is not None else np.array([])])

    # 实际范围（不做等比例缩放）
    x_range = np.nanmax(all_x) - np.nanmin(all_x)
    y_range = np.nanmax(all_y) - np.nanmin(all_y)
    z_range = 40  # 固定 Z 轴范围 40m (±20m)

    # 中心点
    x_mid = (np.nanmax(all_x) + np.nanmin(all_x)) / 2.0
    y_mid = (np.nanmax(all_y) + np.nanmin(all_y)) / 2.0

    box_aspect = (x_range, y_range, z_range)

    # 提取原始点云的 RCS 和 doppler
    ori_rcs = original_data['RCS']
    ori_doppler = original_data['doppler']

    # 定义 RCS 颜色范围（根据数据动态调整）
    rcs_min = np.min(ori_rcs) if final_rcs is None else min(np.min(ori_rcs), np.min(final_rcs))
    rcs_max = np.max(ori_rcs) if final_rcs is None else max(np.max(ori_rcs), np.max(final_rcs))

    # 定义 Doppler 颜色范围（对称，以0为中心）
    doppler_abs_max = np.max(np.abs(ori_doppler)) if final_doppler is None else max(np.max(np.abs(ori_doppler)), np.max(np.abs(final_doppler)))
    doppler_min = -doppler_abs_max
    doppler_max = doppler_abs_max

    # ============ 第一行: RCS 颜色编码 ============

    # 1. 原始点云 - RCS 颜色编码
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(ori_x, ori_y, ori_z, c=ori_rcs, cmap='jet', s=0.3, alpha=0.8,
                           vmin=rcs_min, vmax=rcs_max)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Original - RCS ({len(original_data)} pts)')
    ax1.set_xlim(x_mid - x_range/2, x_mid + x_range/2)
    ax1.set_ylim(y_mid - y_range/2, y_mid + y_range/2)
    ax1.set_zlim(-20, 20)
    ax1.set_box_aspect(box_aspect)
    ax1.view_init(elev=90, azim=180)  # BEV视角：从上往下看
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1)
    cbar1.set_label('RCS (dB)', fontsize=10)

    # 2. 模型输出点云 - RCS 颜色编码（使用原始点云匹配的颜色）
    ax2 = fig.add_subplot(232, projection='3d')
    if sr_x is not None:
        # 模型输出点没有RCS值，用单色显示，但用更明显的颜色
        ax2.scatter(sr_x, sr_y, sr_z, c='lime', s=0.3, alpha=0.8,
                    label=f'Model Output ({len(sr_x)} pts)')
        ax2.set_title(f'Model Output ({len(sr_x)} pts)')
    else:
        ax2.set_title('Model Output (No points)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_xlim(x_mid - x_range/2, x_mid + x_range/2)
    ax2.set_ylim(y_mid - y_range/2, y_mid + y_range/2)
    ax2.set_zlim(-20, 20)
    ax2.set_box_aspect(box_aspect)
    ax2.view_init(elev=90, azim=180)  # BEV视角

    # 3. 过滤后点云 - RCS 颜色编码
    ax3 = fig.add_subplot(233, projection='3d')
    if final_x is not None:
        scatter3 = ax3.scatter(final_x, final_y, final_z, c=final_rcs, cmap='jet', s=0.3, alpha=0.8,
                               vmin=rcs_min, vmax=rcs_max)
        ax3.set_title(f'Filtered - RCS ({len(final_x)} pts)')
        cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.6, pad=0.1)
        cbar3.set_label('RCS (dB)', fontsize=10)
    else:
        ax3.set_title('Filtered (No points)')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_xlim(x_mid - x_range/2, x_mid + x_range/2)
    ax3.set_ylim(y_mid - y_range/2, y_mid + y_range/2)
    ax3.set_zlim(-20, 20)
    ax3.set_box_aspect(box_aspect)
    ax3.view_init(elev=90, azim=180)  # BEV视角

    # ============ 第二行: Doppler 颜色编码 ============

    # 4. 原始点云 - Doppler 颜色编码
    ax4 = fig.add_subplot(234, projection='3d')
    # 使用 RdYlBu_r colormap: 红(正值/接近) -> 黄(零速度) -> 蓝(负值/远离)
    scatter4 = ax4.scatter(ori_x, ori_y, ori_z, c=ori_doppler, cmap='RdYlBu_r', s=0.3, alpha=0.8,
                           vmin=doppler_min, vmax=doppler_max)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title(f'Original - Doppler ({len(original_data)} pts)')
    ax4.set_xlim(x_mid - x_range/2, x_mid + x_range/2)
    ax4.set_ylim(y_mid - y_range/2, y_mid + y_range/2)
    ax4.set_zlim(-20, 20)
    ax4.set_box_aspect(box_aspect)
    ax4.view_init(elev=90, azim=180)  # BEV视角
    cbar4 = plt.colorbar(scatter4, ax=ax4, shrink=0.6, pad=0.1)
    cbar4.set_label('Doppler (m/s)', fontsize=10)

    # 5. 模型输出点云 - Doppler 颜色编码（单色）
    ax5 = fig.add_subplot(235, projection='3d')
    if sr_x is not None:
        ax5.scatter(sr_x, sr_y, sr_z, c='cyan', s=0.3, alpha=0.8)
        ax5.set_title(f'Model Output ({len(sr_x)} pts)')
    else:
        ax5.set_title('Model Output (No points)')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_zlabel('Z (m)')
    ax5.set_xlim(x_mid - x_range/2, x_mid + x_range/2)
    ax5.set_ylim(y_mid - y_range/2, y_mid + y_range/2)
    ax5.set_zlim(-20, 20)
    ax5.set_box_aspect(box_aspect)
    ax5.view_init(elev=90, azim=180)  # BEV视角

    # 6. 过滤后点云 - Doppler 颜色编码
    ax6 = fig.add_subplot(236, projection='3d')
    if final_x is not None:
        scatter6 = ax6.scatter(final_x, final_y, final_z, c=final_doppler, cmap='RdYlBu_r', s=0.3, alpha=0.8,
                               vmin=doppler_min, vmax=doppler_max)
        ax6.set_title(f'Filtered - Doppler ({len(final_x)} pts)')
        cbar6 = plt.colorbar(scatter6, ax=ax6, shrink=0.6, pad=0.1)
        cbar6.set_label('Doppler (m/s)', fontsize=10)
    else:
        ax6.set_title('Filtered (No points)')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_zlabel('Z (m)')
    ax6.set_xlim(x_mid - x_range/2, x_mid + x_range/2)
    ax6.set_ylim(y_mid - y_range/2, y_mid + y_range/2)
    ax6.set_zlim(-20, 20)
    ax6.set_box_aspect(box_aspect)
    ax6.view_init(elev=90, azim=180)  # BEV视角

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  可视化已保存: {save_path}")
        plt.close(fig)
    else:
        # 交互式模式：阻塞显示，用户关闭窗口后继续
        plt.show(block=True)
        plt.close(fig)


def visualize_point_clouds_doppler(original_data: np.ndarray,
                                   sr_points_np: np.ndarray,
                                   valid_points: List[Dict],
                                   sr_rate1: int = 4,
                                   sr_rate2: int = 2,
                                   title: str = "Point Cloud - Doppler Visualization",
                                   save_path: Optional[str] = None):
    """
    可视化点云：专注于 Doppler 颜色编码的详细视图

    参数:
        original_data: 原始点云数据（结构化数组）
        sr_points_np: 模型输出的超分点 [N, 3] (RangeBin, AziBin, EleBin)
        valid_points: 匹配过滤后的有效点列表
        sr_rate1: Range/Azi超分倍数
        sr_rate2: Ele超分倍数
        title: 图像标题
        save_path: 保存路径（可选，不保存则直接显示）
    """
    # 创建 1x3 的子图布局
    fig = plt.figure(figsize=(18, 6))

    # 计算所有点的坐标范围
    ori_x, ori_y, ori_z = calculate_original_coordinates(original_data)

    # 计算过滤后点坐标
    final_x, final_y, final_z, final_doppler = None, None, None, None
    if len(valid_points) > 0:
        final_x = np.array([p['SR_x'] for p in valid_points])
        final_y = np.array([p['SR_y'] for p in valid_points])
        final_z = np.array([p['SR_z'] for p in valid_points])
        final_doppler = np.array([p['doppler'] for p in valid_points])

    # 统一坐标范围
    all_x = np.concatenate([ori_x, final_x if final_x is not None else np.array([])])
    all_y = np.concatenate([ori_y, final_y if final_y is not None else np.array([])])

    x_range = np.nanmax(all_x) - np.nanmin(all_x)
    y_range = np.nanmax(all_y) - np.nanmin(all_y)
    z_range = 40

    x_mid = (np.nanmax(all_x) + np.nanmin(all_x)) / 2.0
    y_mid = (np.nanmax(all_y) + np.nanmin(all_y)) / 2.0

    box_aspect = (x_range, y_range, z_range)

    # Doppler 颜色范围（对称）
    ori_doppler = original_data['doppler']
    doppler_abs_max = np.max(np.abs(ori_doppler)) if final_doppler is None else max(np.max(np.abs(ori_doppler)), np.max(np.abs(final_doppler)))
    doppler_min = -doppler_abs_max
    doppler_max = doppler_abs_max

    # 1. 原始点云 - Doppler 颜色编码
    ax1 = fig.add_subplot(131, projection='3d')
    # 使用 RdYlBu_r colormap: 红(正值/接近) -> 黄(零速度) -> 蓝(负值/远离)
    scatter1 = ax1.scatter(ori_x, ori_y, ori_z, c=ori_doppler, cmap='RdYlBu_r', s=0.3, alpha=0.8,
                           vmin=doppler_min, vmax=doppler_max)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Original - Doppler ({len(original_data)} pts)')
    ax1.set_xlim(x_mid - x_range/2, x_mid + x_range/2)
    ax1.set_ylim(y_mid - y_range/2, y_mid + y_range/2)
    ax1.set_zlim(-20, 20)
    ax1.set_box_aspect(box_aspect)
    ax1.view_init(elev=90, azim=180)  # BEV视角：从上往下看
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1)
    cbar1.set_label('Doppler (m/s)', fontsize=10)

    # 2. 过滤后点云 - Doppler 颜色编码
    ax2 = fig.add_subplot(132, projection='3d')
    if final_x is not None:
        scatter2 = ax2.scatter(final_x, final_y, final_z, c=final_doppler, cmap='RdYlBu_r', s=0.3, alpha=0.8,
                               vmin=doppler_min, vmax=doppler_max)
        ax2.set_title(f'Filtered - Doppler ({len(final_x)} pts)')
    else:
        ax2.set_title('Filtered - Doppler (No points)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_xlim(x_mid - x_range/2, x_mid + x_range/2)
    ax2.set_ylim(y_mid - y_range/2, y_mid + y_range/2)
    ax2.set_zlim(-20, 20)
    ax2.set_box_aspect(box_aspect)
    ax2.view_init(elev=90, azim=180)  # BEV视角
    if final_x is not None:
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.1)
        cbar2.set_label('Doppler (m/s)', fontsize=10)

    # 3. Doppler 值分布对比
    ax3 = fig.add_subplot(133)
    if final_doppler is not None:
        ax3.hist(ori_doppler, bins=50, alpha=0.5, label='Original', color='blue', density=True)
        ax3.hist(final_doppler, bins=50, alpha=0.5, label='Filtered', color='red', density=True)
        ax3.set_xlabel('Doppler (m/s)')
        ax3.set_ylabel('Density')
        ax3.set_title('Doppler Distribution')
        ax3.legend()
        ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Doppler可视化已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show(block=True)
        plt.close(fig)


def visualize_aai_comparison(
    sr_xyz: np.ndarray,
    aai_xyz: np.ndarray,
    merged_xyz: np.ndarray,
    occ_gt_xyz: Optional[np.ndarray] = None,
    grid_res: float = GRID_RES,
    title: str = "AAI Comparison Visualization",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12),
    merge_stats: Optional[Dict] = None,
    aai_dynamic_mask: Optional[np.ndarray] = None,
    original_xyz: Optional[np.ndarray] = None,
    original_dynamic_mask: Optional[np.ndarray] = None
):
    """
    AAI叠加结果可视化 - 展示SR、AAI、合并后的点云与真值对比（BEV视图）

    当提供 aai_dynamic_mask 时，AAI点云会按动态/静态分别着色，便于观察
    动态点云的空间分布。

    Args:
        sr_xyz: 超分点云xyz坐标 [N, 3]
        aai_xyz: AAI处理后的xyz坐标 [M, 3]
        merged_xyz: 合并后的xyz坐标 [K, 3]
        occ_gt_xyz: Occ真值xyz坐标 [可选]仅用于对比子图的命中分析
        grid_res: 网格分辨率
        title: 图像标题
        save_path: 保存路径
        figsize: 图像大小
        merge_stats: 合并统计信息
        aai_dynamic_mask: AAI点的动态/静态标记布尔数组 [M]，True表示动态点。
                          仅在使用动静态分离AAI处理时有效，None时全部视为静态。
        original_xyz: 原始雷达点云xyz坐标 [P, 3] [可选]用于子图4展示原始点云全部显示为静态。
        original_dynamic_mask: 原始点云的动态/静态标记布尔数组 [P]，True表示动态点。
                               仅在使用动静态分离AAI处理时有效，None时原始点云全部显示为静态。
    """
    # 投影到BEV（忽略Z维度）
    def project_to_bev_set(xyz):
        if xyz.shape[0] == 0:
            return set()
        bev_set = set()
        grid_indices = xyz_to_grid_index(xyz, grid_res)
        for idx in grid_indices:
            bev_set.add((idx[0], idx[1]))  # (GridY, GridX)
        return bev_set

    # 分割AAI点云为静态和动态（用于差异化着色）
    if aai_dynamic_mask is not None and aai_xyz.shape[0] > 0 \
            and len(aai_dynamic_mask) == aai_xyz.shape[0]:
        aai_static_xyz = aai_xyz[~aai_dynamic_mask]
        aai_dynamic_xyz = aai_xyz[aai_dynamic_mask]
    else:
        aai_static_xyz = aai_xyz
        aai_dynamic_xyz = np.empty((0, 3), dtype=np.float32)

    # 分割原始点云为静态和动态（用于子图4上色）
    if original_xyz is not None and original_dynamic_mask is not None \
            and original_xyz.shape[0] > 0 \
            and len(original_dynamic_mask) == original_xyz.shape[0]:
        original_static_xyz = original_xyz[~original_dynamic_mask]
        original_dynamic_xyz = original_xyz[original_dynamic_mask]
    elif original_xyz is not None:
        original_static_xyz = original_xyz
        original_dynamic_xyz = np.empty((0, 3), dtype=np.float32)
    else:
        original_static_xyz = np.empty((0, 3), dtype=np.float32)
        original_dynamic_xyz = np.empty((0, 3), dtype=np.float32)

    sr_bev = project_to_bev_set(sr_xyz)
    aai_static_bev = project_to_bev_set(aai_static_xyz)
    aai_dynamic_bev = project_to_bev_set(aai_dynamic_xyz)
    aai_bev = aai_static_bev | aai_dynamic_bev
    merged_bev = project_to_bev_set(merged_xyz)
    gt_bev = project_to_bev_set(occ_gt_xyz) if occ_gt_xyz is not None else set()
    original_static_bev = project_to_bev_set(original_static_xyz)
    original_dynamic_bev = project_to_bev_set(original_dynamic_xyz)
    original_bev = original_static_bev | original_dynamic_bev

    # 计算BEV范围
    all_voxels = sr_bev | aai_bev | merged_bev | gt_bev | original_bev
    if len(all_voxels) == 0:
        print("警告：无有效网格数据，跳过AAI可视化")
        return

    all_y = [v[0] for v in all_voxels]
    all_x = [v[1] for v in all_voxels]

    y_min, y_max = min(all_y), max(all_y)
    x_min, x_max = min(all_x), max(all_x)

    # 扩展范围
    y_range = y_max - y_min + 2
    x_range = x_max - x_min + 2

    extent = [x_min*grid_res, (x_max+1)*grid_res, y_min*grid_res, (y_max+1)*grid_res]

    # 创建 2x3 子图布局
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # ===== 子图1: SR超分网格 =====
    ax1 = axes[0, 0]
    sr_grid = np.zeros((y_range, x_range), dtype=np.uint8)
    for grid_y, grid_x in sr_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            sr_grid[local_y, local_x] = 1
    ax1.imshow(sr_grid, cmap='Blues', origin='lower', extent=extent)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'SR Point Cloud\n{len(sr_bev)} cells')
    ax1.grid(True, alpha=0.3)

    # ===== 子图2: AAI网格（区分静态/动态） =====
    ax2 = axes[0, 1]
    aai_rgb_grid = np.ones((y_range, x_range, 3), dtype=np.float32)
    color_aai_static = np.array([0.91, 0.60, 0.20])    # 橙色（静态）
    color_aai_dynamic = np.array([0.91, 0.24, 0.68])   # 品红色（动态）

    for grid_y, grid_x in aai_static_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            aai_rgb_grid[local_y, local_x] = color_aai_static

    for grid_y, grid_x in aai_dynamic_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            aai_rgb_grid[local_y, local_x] = color_aai_dynamic

    ax2.imshow(aai_rgb_grid, origin='lower', extent=extent)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'AAI Point Cloud\n{len(aai_bev)} cells (Static: {len(aai_static_bev)}, Dynamic: {len(aai_dynamic_bev)})')
    aai_legend = [
        mpatches.Patch(facecolor='#E8993C', label=f'Static ({len(aai_static_bev)})'),
        mpatches.Patch(facecolor='#E83DAE', label=f'Dynamic ({len(aai_dynamic_bev)})'),
    ]
    ax2.legend(handles=aai_legend, loc='upper right', fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ===== 子图3: 合并后网格（区分AAI静态/动态叠加部分） =====
    ax3 = axes[0, 2]

    # 直接在BEV层面计算各类别
    sr_only_bev = sr_bev - aai_bev
    aai_added_static_bev = aai_static_bev - sr_bev
    aai_added_dynamic_bev = aai_dynamic_bev - sr_bev
    overlap_bev = sr_bev & aai_bev

    # 使用RGB颜色值填充
    merged_rgb_grid = np.ones((y_range, x_range, 3), dtype=np.float32)

    color_blue = np.array([0.20, 0.60, 0.86])
    color_red = np.array([0.91, 0.30, 0.24])       # AAI静态新增
    color_magenta = np.array([0.91, 0.24, 0.68])   # AAI动态新增
    color_green = np.array([0.18, 0.80, 0.44])

    for grid_y, grid_x in sr_only_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            merged_rgb_grid[local_y, local_x] = color_blue

    for grid_y, grid_x in aai_added_static_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            merged_rgb_grid[local_y, local_x] = color_red

    for grid_y, grid_x in aai_added_dynamic_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            merged_rgb_grid[local_y, local_x] = color_magenta

    for grid_y, grid_x in overlap_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            merged_rgb_grid[local_y, local_x] = color_green

    ax3.imshow(merged_rgb_grid, origin='lower', extent=extent)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f'Merged (SR+AAI)\n{len(merged_bev)} cells')

    merged_legend = [
        mpatches.Patch(facecolor='#3498DB', label='SR only'),
        mpatches.Patch(facecolor='#E74C3C', label='AAI static added'),
        mpatches.Patch(facecolor='#E83DAE', label='AAI dynamic added'),
        mpatches.Patch(facecolor='#2ECC71', label='Overlap'),
    ]
    ax3.legend(handles=merged_legend, loc='upper right', fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ===== 子图4: 原始点云网格（区分静态/动态） =====
    ax4 = axes[1, 0]
    original_rgb_grid = np.ones((y_range, x_range, 3), dtype=np.float32)
    color_ori_static = np.array([0.50, 0.50, 0.50])    # 灰色（静态）
    color_ori_dynamic = np.array([0.91, 0.24, 0.68])   # 品红色（动态）

    for grid_y, grid_x in original_static_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            original_rgb_grid[local_y, local_x] = color_ori_static

    for grid_y, grid_x in original_dynamic_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            original_rgb_grid[local_y, local_x] = color_ori_dynamic

    ax4.imshow(original_rgb_grid, origin='lower', extent=extent)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title(f'Original Point Cloud\n{len(original_bev)} cells (Static: {len(original_static_bev)}, Dynamic: {len(original_dynamic_bev)})')
    ori_legend = [
        mpatches.Patch(facecolor='#808080', label=f'Static ({len(original_static_bev)})'),
        mpatches.Patch(facecolor='#E83DAE', label=f'Dynamic ({len(original_dynamic_bev)})'),
    ]
    ax4.legend(handles=ori_legend, loc='upper right', fontsize=7)
    ax4.grid(True, alpha=0.3)

    # ===== 子图5: 叠加对比（区分AAI静态/动态） =====
    ax5 = axes[1, 1]
    hit_bev = merged_bev & gt_bev
    combined_rgb_grid = np.ones((y_range, x_range, 3), dtype=np.float32)

    color_gt_missed = np.array([0.61, 0.36, 0.71])     # 紫色
    color_sr_only_cmp = np.array([0.20, 0.60, 0.86])   # 蓝色
    color_aai_static_cmp = np.array([0.90, 0.47, 0.13])  # 橙色
    color_aai_dynamic_cmp = np.array([0.91, 0.24, 0.68]) # 品红色
    color_overlap_cmp = np.array([0.95, 0.61, 0.07])    # 橙黄
    color_hit_cmp = np.array([0.18, 0.80, 0.44])        # 绿色

    for grid_y, grid_x in gt_bev - merged_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            combined_rgb_grid[local_y, local_x] = color_gt_missed

    for grid_y, grid_x in sr_bev - aai_bev - gt_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            combined_rgb_grid[local_y, local_x] = color_sr_only_cmp

    for grid_y, grid_x in aai_static_bev - sr_bev - gt_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            combined_rgb_grid[local_y, local_x] = color_aai_static_cmp

    for grid_y, grid_x in aai_dynamic_bev - sr_bev - gt_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            combined_rgb_grid[local_y, local_x] = color_aai_dynamic_cmp

    for grid_y, grid_x in (sr_bev & aai_bev) - gt_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            combined_rgb_grid[local_y, local_x] = color_overlap_cmp

    for grid_y, grid_x in hit_bev:
        local_y = grid_y - y_min
        local_x = grid_x - x_min
        if 0 <= local_y < y_range and 0 <= local_x < x_range:
            combined_rgb_grid[local_y, local_x] = color_hit_cmp

    ax5.imshow(combined_rgb_grid, origin='lower', extent=extent)
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title(f'Comparison\nHit: {len(hit_bev)} cells')
    ax5.grid(True, alpha=0.3)

    legend_elements = [
        mpatches.Patch(facecolor='#9B59B6', label='GT missed'),
        mpatches.Patch(facecolor='#3498DB', label='SR only'),
        mpatches.Patch(facecolor='#E67E22', label='AAI static only'),
        mpatches.Patch(facecolor='#E83DAE', label='AAI dynamic only'),
        mpatches.Patch(facecolor='#F39C12', label='Overlap'),
        mpatches.Patch(facecolor='#2ECC71', label='Hit'),
    ]
    ax5.legend(handles=legend_elements, loc='upper right', fontsize=7)

    # ===== 子图6: 统计信息 =====
    ax6 = axes[1, 2]
    ax6.axis('off')

    overlap_count = len(sr_bev & aai_bev)
    sr_only_count = len(sr_bev - aai_bev)
    aai_static_added_count = len(aai_added_static_bev)
    aai_dynamic_added_count = len(aai_added_dynamic_bev)
    hit_count = len(hit_bev)

    stats_text = f"""
AAI Comparison Statistics
=========================

Point Cloud Counts:
- Original cells: {len(original_bev)} (Static: {len(original_static_bev)}, Dynamic: {len(original_dynamic_bev)})
- SR cells: {len(sr_bev)}
- AAI cells: {len(aai_bev)} (Static: {len(aai_static_bev)}, Dynamic: {len(aai_dynamic_bev)})
- Merged cells: {len(merged_bev)}
- GT cells: {len(gt_bev)}

Merge Details:
- Overlap (SR∩AAI): {overlap_count}
- SR only: {sr_only_count}
- AAI static added: {aai_static_added_count}
- AAI dynamic added: {aai_dynamic_added_count}

Hit Analysis:
- Hit cells: {hit_count}
- GT coverage: {hit_count/len(gt_bev) if len(gt_bev) > 0 else 0:.2%}
"""

    if merge_stats:
        stats_text += f"""
Bin-level Statistics:
- SR points: {merge_stats.get('sr_points', 0)}
- AAI points: {merge_stats.get('aai_points', 0)}
- Overlap bins: {merge_stats.get('overlap_bins', 0)}
"""

    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
        print(f"  AAI对比可视化已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show(block=True)
        plt.close(fig)


def calculate_sr_coordinates(sr_range_bin: int, sr_azi_bin: int, sr_ele_bin: int,
                             sr_rate1: int = 4, sr_rate2: int = 2) -> Optional[Tuple]:
    """
    根据超分bin值计算坐标

    参数:
        sr_range_bin: 超分RangeBin (0-2047)
        sr_azi_bin: 超分AziBin (0-511)
        sr_ele_bin: 超分EleBin (0-63)
        sr_rate1: Range和Azimuth超分倍数
        sr_rate2: Elevation超分倍数

    返回:
        (SR_x, SR_y, SR_z, SR_range, SR_sinAzi, SR_sinEle) 或 None(无效点)
    """
    # 系数参数（按用户指定）
    range_coef_metadata = 40960
    azimuth_coef_metadata = 1047
    elevation_coef_metadata = 1397
    range_zoom = 3

    # 计算系数
    range_coef = range_coef_metadata / 65536.0
    azimuth_coef = azimuth_coef_metadata / 65536.0
    elevation_coef = elevation_coef_metadata / 65536.0

    sr_range_coef = range_coef / sr_rate1
    sr_azimuth_coef = azimuth_coef / sr_rate1
    sr_elevation_coef = elevation_coef / sr_rate2

    # 半宽参数
    azi_half = 64
    ele_half = 16
    sr_azi_half = azi_half * sr_rate1
    sr_ele_half = ele_half * sr_rate2

    # 计算range和角度
    SR_range = sr_range_coef * (sr_range_bin - 0.5 * range_zoom)
    SR_sinAzi = (sr_azi_bin * (-1) + sr_azi_half) * sr_azimuth_coef
    SR_sinEle = (sr_ele_bin * (-1) + sr_ele_half) * sr_elevation_coef

    # 计算笛卡尔坐标（原始）
    SR_x_Ori = SR_range * SR_sinAzi
    SR_z = SR_range * SR_sinEle

    # 计算y，检查几何有效性
    y_squared = SR_range**2 - SR_x_Ori**2 - SR_z**2
    if y_squared <= 0:
        return None  # 无效点：y平方为零或负值

    SR_y_Ori = math.sqrt(y_squared)

    # 坐标系变换（按用户指定）
    SR_x = SR_y_Ori
    SR_y = SR_x_Ori * (-1)

    return SR_x, SR_y, SR_z, SR_range, SR_sinAzi, SR_sinEle


def is_valid_fov(sinAzi: float, sinEle: float) -> bool:
    """
    检查点是否在有效视场角范围内

    FOV: 水平角 ±50°，俯仰角 ±20°

    参数:
        sinAzi: 方位角sin值
        sinEle: 俯仰角sin值

    返回:
        True表示有效，False表示超出FOV
    """
    azi_threshold = math.sin(math.radians(50))  # ≈ 0.766
    ele_threshold = math.sin(math.radians(20))  # ≈ 0.342

    return abs(sinAzi) <= azi_threshold and abs(sinEle) <= ele_threshold


# ==================== 点云匹配函数 ====================
# 预计单搜索顺序：按照由小到大排序
# 搜索范围：range ±2, azi ±2, ele ±1（共5×5×3=75个位置）
# 以中心点(0,0,0)为原点，计算每个偏移位置的距离并排序
def _generate_distance_sorted_offsets():
    """
    生成按距离排序的搜索偏移量列表

    返回: [(range_offset, azi_offset, ele_offset, distance), ...] 按距离升序排列
    """
    offsets = []
    for r_off in range(-2, 3):  # -2, -1, 0, 1, 2
        for a_off in range(-2, 3):  # -2, -1, 0, 1, 2
            for e_off in range(-1, 2):  # -1, 0, 1
                # 计算距离（简单欧式）
                distance = math.sqrt(r_off**2 + a_off**2 + e_off**2)
                offsets.append((r_off, a_off, e_off, distance))

    # 按距离升序排序
    offsets.sort(key=lambda x: x[3])
    return offsets

# 预计算的搜索顺序（全局常量）
_SEARCH_OFFSETS_SORTED = _generate_distance_sorted_offsets()

# Numba JIT 加速的核心搜索函数（按距离优先顺序搜索）
@jit(nopython=True, cache=True)
def _search_candidates_numba_distance_first(sr_range_bin, sr_azi_bin, sr_ele_bin,
                                             range_bins, azi_bins, ele_bins,
                                             point_index_map, sr_rate1, sr_rate2,
                                             range_dim, azi_dim, ele_dim,
                                             search_offsets):
    """
    Numba JIT 加速的搜索函数 - 按距离优先顺序搜索

    搜索顺序已预先按距离排序，找到的第一个满足条件的点即为距离最近的点

    返回: candidate_indices 列表（第一个元素为距离最近的匹配点）
    """
    # 计算基准 bin（原始分辨率）
    base_range_bin = sr_range_bin // sr_rate1
    base_azi_bin = sr_azi_bin // sr_rate1
    base_ele_bin = sr_ele_bin // sr_rate2

    # 阈值
    range_threshold = sr_rate1 * 2
    azi_threshold = sr_rate1 * 2
    ele_threshold = sr_rate2

    # 按预定义的距离顺序搜索
    candidate_indices = []

    for i in range(len(search_offsets)):
        r_off = int(search_offsets[i][0])
        a_off = int(search_offsets[i][1])
        e_off = int(search_offsets[i][2])

        # 计算实际搜索位置
        r = base_range_bin + r_off
        a = base_azi_bin + a_off
        e = base_ele_bin + e_off

        # 检查边界
        if r < 0 or r >= range_dim or a < 0 or a >= azi_dim or e < 0 or e >= ele_dim:
            continue

        # 查找该位置的点索引
        idx = point_index_map[r, a, e]
        if idx >= 0:
            # 验证阈值条件（与原逻辑一致）
            scaled_range = range_bins[idx] * sr_rate1
            scaled_azi = azi_bins[idx] * sr_rate1
            scaled_ele = ele_bins[idx] * sr_rate2

            diff_range = abs(sr_range_bin - scaled_range)
            diff_azi = abs(sr_azi_bin - scaled_azi)
            diff_ele = abs(sr_ele_bin - scaled_ele)

            if diff_range < range_threshold and \
               diff_azi < azi_threshold and \
               diff_ele <= ele_threshold:
                candidate_indices.append(idx)
                # 找到第一个匹配点后继续搜索，收集所有候选
                # 但由于已按距离排序，第一个即为最近的

    return candidate_indices


class PointCloudLookupTable:
    """
    原始点云查找表（空间换时间策略 + Numba JIT 加速）

    将原始点云按 (RangeBin, AziBin, EleBin) 映射到固定大小的内存空间，
    支持快速查找匹配点。

    空间大小: [range_dim, azi_dim, ele_dim] = [512, 128, 32]
    """

    def __init__(self, original_data: np.ndarray,
                 range_dim: int = 512, azi_dim: int = 128, ele_dim: int = 32):
        """
        构建查找表

        参数:
            original_data: 原始点云数据（结构化数组）
            range_dim: Range维度大小
            azi_dim: Azimuth维度大小
            ele_dim: Elevation维度大小
        """
        self.range_dim = range_dim
        self.azi_dim = azi_dim
        self.ele_dim = ele_dim

        # 提取 bin 值
        self.range_bins = original_data['RangeBin'].astype(np.int32)
        self.azi_bins = original_data['AziBin'].astype(np.int32)
        self.ele_bins = original_data['EleBin'].astype(np.int32)

        # 裁剪到有效范围
        self.range_bins = np.clip(self.range_bins, 0, range_dim - 1)
        self.azi_bins = np.clip(self.azi_bins, 0, azi_dim - 1)
        self.ele_bins = np.clip(self.ele_bins, 0, ele_dim - 1)

        # 存储原始数据引用
        self.original_data = original_data

        # 构建三维 numpy 查找表（用于 numba JIT）
        # 每个位置存储一个点索引，-1 表示无点
        # 注意：同一位置多个点只保留最后一个（反向遍历）
        self.point_index_map = np.full((range_dim, azi_dim, ele_dim), -1, dtype=np.int32)
        for idx in range(len(original_data) - 1, -1, -1):
            r, a, e = self.range_bins[idx], self.azi_bins[idx], self.ele_bins[idx]
            self.point_index_map[r, a, e] = idx

        # 同时保留字典查找表（用于处理同一位置多个点的情况）
        self.lookup_table = {}
        for idx in range(len(original_data)):
            key = (self.range_bins[idx], self.azi_bins[idx], self.ele_bins[idx])
            if key not in self.lookup_table:
                self.lookup_table[key] = []
            self.lookup_table[key].append(idx)

        # 使用 numba 加速的标志
        self.use_numba = NUMBA_AVAILABLE

        # 预计算搜索偏移量数组（用于 numba）
        # 格式: numpy array [N, 4] = [range_off, azi_off, ele_off, distance]
        self.search_offsets_array = np.array(
            [[o[0], o[1], o[2], o[3]] for o in _SEARCH_OFFSETS_SORTED],
            dtype=np.float64
        )

    def find_matches(self, sr_range_bin: int, sr_azi_bin: int, sr_ele_bin: int,
                     sr_rate1: int = 4, sr_rate2: int = 2) -> List[int]:
        """
        根据超分 bin 值查找匹配的原始点索引（距离优先匹配）

        搜索范围：5×5×3 = 75 个位置
        搜索顺序：按距离从小到大（预计算），找到的第一个匹配点即为最近点

        匹配条件:
        - |SR_RangeBin - sr_rate1 * RangeBin| < sr_rate1 * 2 (即 < 8)
        - |SR_AziBin - sr_rate1 * AziBin| < sr_rate1 * 2 (即 < 8)
        - |SR_EleBin - sr_rate2 * EleBin| <= sr_rate2 (即 <= 2)

        参数:
            sr_range_bin: 超分 RangeBin
            sr_azi_bin: 超分 AziBin
            sr_ele_bin: 超分 EleBin
            sr_rate1: Range/Azi 超分倍数
            sr_rate2: Ele 超分倍数

        返回:
            匹配的原始点索引列表（第一个元素为距离最近的匹配点）
        """
        if self.use_numba:
            # 使用 Numba JIT 加速（距离优先搜索）
            candidate_indices = _search_candidates_numba_distance_first(
                sr_range_bin, sr_azi_bin, sr_ele_bin,
                self.range_bins, self.azi_bins, self.ele_bins,
                self.point_index_map, sr_rate1, sr_rate2,
                self.range_dim, self.azi_dim, self.ele_dim,
                self.search_offsets_array
            )

            # 处理返回结果（numba 可能返回 typed list 或 Python list）
            if len(candidate_indices) > 0:
                # 如果是 numba typed list，需要转换；如果是普通 list，直接返回
                if hasattr(candidate_indices, 'tolist'):
                    return candidate_indices.tolist()
                else:
                    return candidate_indices
            return []

        else:
            # 无 numba 时使用字典查找表（距离优先搜索）
            base_range_bin = sr_range_bin // sr_rate1
            base_azi_bin = sr_azi_bin // sr_rate1
            base_ele_bin = sr_ele_bin // sr_rate2

            # 阈值
            range_threshold = sr_rate1 * 2
            azi_threshold = sr_rate1 * 2
            ele_threshold = sr_rate2

            candidate_indices = []

            # 按预定义的距离顺序搜索
            for r_off, a_off, e_off, _ in _SEARCH_OFFSETS_SORTED:
                r = base_range_bin + r_off
                a = base_azi_bin + a_off
                e = base_ele_bin + e_off

                # 检查边界
                if r < 0 or r >= self.range_dim or \
                   a < 0 or a >= self.azi_dim or \
                   e < 0 or e >= self.ele_dim:
                    continue

                key = (r, a, e)
                if key in self.lookup_table:
                    indices = self.lookup_table[key]
                    for idx in indices:
                        scaled_range = self.range_bins[idx] * sr_rate1
                        scaled_azi = self.azi_bins[idx] * sr_rate1
                        scaled_ele = self.ele_bins[idx] * sr_rate2

                        diff_range = abs(sr_range_bin - scaled_range)
                        diff_azi = abs(sr_azi_bin - scaled_azi)
                        diff_ele = abs(sr_ele_bin - scaled_ele)

                        if diff_range < range_threshold and \
                           diff_azi < azi_threshold and \
                           diff_ele <= ele_threshold:
                            candidate_indices.append(idx)
                            # 找到第一个匹配点后继续搜索，收集所有候选
                            # 但由于已按距离排序，第一个即为最近的

            return candidate_indices

    def get_point_data(self, idx: int) -> Dict:
        """
        获取指定索引点的属性数据

        参数:
            idx: 原始点索引

        返回:
            点属性字典
        """
        return {
            'RCS': self.original_data['RCS'][idx],
            'doppler': self.original_data['doppler'][idx],
            'AbsV': self.original_data['AbsV'][idx],
            'Vx': self.original_data['Vx'][idx],
            'Vy': self.original_data['Vy'][idx],
            'PowerBin': int(self.original_data['PowerBin'][idx]),
            'PhaseBin': int(self.original_data['PhaseBin'][idx]),
            'power': self.original_data['power'][idx],
            'DopplerBin': int(self.original_data['DopplerBin'][idx])
        }


def _reuse_count_to_db_attenuation(reuse_count: int) -> float:
    """
    根据原始点被复用的次数计算RCS衰减(dB)，查表近似 -10*log10(N)。

    能量均分模型：同一原始点被 N 个SR点复用时，功率或总散射能量被放大。
    每个SR点分得 1/N 的能量，对应衰减 -10*log10(N) dB。查表近似（向下取整到
    最近的2的幂对应的桶）：

        1 次:  0 dB  (线性因子 1.000)
        2 次: -3 dB  (线性因子 0.501)
        4 次: -6 dB  (线性因子 0.251)
        8 次: -9 dB  (线性因子 0.126)
       16次: -12 dB (线性因子 0.063)
        ...

    非整数2的幂的复用次数向下取整到最近的2的幂对应桶：
        3 次 -> -3dB, 5~7 次 -> -6dB, 9~15 次 -> -9dB ...

    参数:
        reuse_count: 原始点被SR点匹配的总次数 (>=0)
    返回:
        衰减值(dB)，<=0；reuse_count<=1时返回0
    """
    if reuse_count <= 1:
        return 0.0
    # 每多一倍复用衰减 -3 dB (≈ -10*log10(2))
    # level: 2->1, 3->1, 4->2, 7->2, 8->3, 15->3, 16->4 ...
    level = reuse_count.bit_length() - 1
    return -3.0 * level


def match_and_assign_values(sr_points: np.ndarray, original_data: np.ndarray,
                            sr_rate1: int = 4, sr_rate2: int = 2,
                            max_match_per_orig: int = 0,
                            is_sr_arr: Optional[np.ndarray] = None,
                            enable_reuse_decay: bool = True) -> List[Dict]:
    """
    将超分点云与原始点云匹配，赋值属性（优化版本：空间换时间）

    使用查找表策略:
    1. 将原始点云按 (RangeBin, AziBin, EleBin) 映射到查找表
    2. 超分点计算对应的原始 bin 范围，直接查找
    3. 复杂度从 O(N*M) 降为 O(N + M)

    匹配条件:
    - |SR_RangeBin - sr_rate1 * RangeBin| < sr_rate1 * 2 (即 < 8)
    - |SR_AziBin - sr_rate1 * AziBin| < sr_rate1 * 2 (即 < 8)
    - |SR_EleBin - sr_rate2 * EleBin| <= sr_rate2 (即 <= 2)

    选择差值和最小的原始点进行赋值

    RCS 赋值策略:
        (1) bin 距离线性衰减（始终启用）：衰减因子 = max(0, 1 - L1_bin_distance / max_L1_distance)
            距离为0时因子1.0，完整RCS；距离最远时因子0.0。
            仅使用整数 L1 距离和一次除法，不需要 exp()/log()，浮点排序。
        (2) 复用次数衰减(能量均分，可选)：同一原始点被 N 个SR点复用时，每个SR点的RCS
            再乘以 10^(reuse_db/10)，其中 reuse_db 由查表得到 (1次:0dB/2次:-3dB/
            4次:-6dB/8次:-9dB...)，使总散射能量不因点数增多而被放大。
            仅当 enable_reuse_decay=True 时启用。
        最终 RCS = 原始RCS × 距离衰减因子 × (复用次数衰减因子)。
    Doppler 等其他属性：直接从最近原始点赋值（速度场局部均匀）。

    两阶段流程:
        Phase 1: 为每个SR点找最近原始点，累计每个原始点的复用次数。
        Phase 2: 基于最终复用次数计算RCS，所有复用同一原始点的SR点使用相同衰减。

    参数:
        sr_points: 超分点数组 [N, 3] (SR_RangeBin, SR_AziBin, SR_EleBin)
        original_data: 原始点云数据
        sr_rate1: Range/Azi超分倍数
        sr_rate2: Ele超分倍数
        max_match_per_orig: 每个原始点最大被匹配次数。
            >0 时启用限制：SR点只取距离最近的原始点，
            若该点已达匹配上限则直接跳过该SR点（不再找其他候选）；
            0 表示不限制（默认）。
        is_sr_arr: 每个超分点的来源标记 [N]，1=模型SR生成，0=AAI叠加的原始点云。
                   None时默认全部为1。该值会附加到匹配结果的 is_sr 字段。
        enable_reuse_decay: 是否启用复用次数衰减（能量均分），默认True。
                            False时RCS仅受bin距离线性衰减，复用同一原始点的SR点
                            各自保留完整距离衰减后的RCS。

    返回:
        匹配后的点列表，包含赋值后的属性
    """
    # 构建查找表
    lookup_table = PointCloudLookupTable(original_data)

    # 原始点被匹配次数计数（用于复用次数衰减）
    match_count = np.zeros(len(original_data), dtype=np.int32)

    # RCS 距离衰减的最大 L1 bin 距离（与匹配阈值上界一致）
    range_thresh = sr_rate1 * 2
    azi_thresh = sr_rate1 * 2
    ele_thresh = sr_rate2
    max_l1_distance = range_thresh + azi_thresh + ele_thresh

    n_skipped_by_limit = 0

    # === Phase 1: 为每个SR点找到最佳匹配的原始点，累计复用次数（先不算RCS） ===
    # pending_matches: (best_match_idx, bin_distance, sr_r, sr_a, sr_e, is_sr_val)
    pending_matches = []

    for i, sr_point in enumerate(sr_points):
        sr_range_bin, sr_azi_bin, sr_ele_bin = int(sr_point[0]), int(sr_point[1]), int(sr_point[2])

        # 使用查找表快速查找匹配点
        matched_indices = lookup_table.find_matches(
            sr_range_bin, sr_azi_bin, sr_ele_bin,
            sr_rate1, sr_rate2
        )

        if len(matched_indices) == 0:
            continue

        # 取距离最近的匹配点（列表已按距离升序排列）
        best_match_idx = matched_indices[0]

        # 限制每个原始点的最大匹配次数：只要最近原始点已满，直接跳过该SR点
        if max_match_per_orig > 0 and match_count[best_match_idx] >= max_match_per_orig:
            n_skipped_by_limit += 1
            continue

        match_count[best_match_idx] += 1

        # 计算 bin 距离（用于距离衰减）
        scaled_range = lookup_table.range_bins[best_match_idx] * sr_rate1
        scaled_azi = lookup_table.azi_bins[best_match_idx] * sr_rate1
        scaled_ele = lookup_table.ele_bins[best_match_idx] * sr_rate2

        diff_range = abs(sr_range_bin - scaled_range)
        diff_azi = abs(sr_azi_bin - scaled_azi)
        diff_ele = abs(sr_ele_bin - scaled_ele)
        # L1 距离（整数，无需浮点排序）
        bin_distance = diff_range + diff_azi + diff_ele

        # 点来源标记：1=模型SR生成，0=AAI叠加的原始点云
        is_sr_val = int(is_sr_arr[i]) if is_sr_arr is not None else 1

        pending_matches.append(
            (best_match_idx, bin_distance, sr_range_bin, sr_azi_bin, sr_ele_bin, is_sr_val)
        )

    # === Phase 2: 基于最终复用次数计算RCS（距离衰减 × 复用次数衰减） ===
    matched_points = []
    reuse_hist = {}  # 复用次数 -> SR点数（用于日志统计）

    for best_match_idx, bin_distance, sr_range_bin, sr_azi_bin, sr_ele_bin, is_sr_val in pending_matches:
        # (1) 距离衰减：距离0 -> 1.0(完整RCS)，距离max -> 0.0
        if max_l1_distance > 0:
            rcs_decay = max(0.0, 1.0 - bin_distance / max_l1_distance)
        else:
            rcs_decay = 1.0

        # (2) 复用次数衰减：N个SR点复用同一原始点时，每个分得 1/N 能量（查表近似）
        # dB -> 线性因子: 10^(dB/10) => 1次:1.0, 2次:0.501, 4次:0.251, 8次:0.126 ...
        reuse_n = int(match_count[best_match_idx])
        if enable_reuse_decay:
            reuse_db = _reuse_count_to_db_attenuation(reuse_n)
            reuse_factor = 10.0 ** (reuse_db / 10.0)
        else:
            reuse_db = 0.0
            reuse_factor = 1.0

        if reuse_n > 1:
            reuse_hist[reuse_n] = reuse_hist.get(reuse_n, 0) + 1

        # 从查找表获取属性
        point_data = lookup_table.get_point_data(best_match_idx)

        # 创建匹配点：RCS = 原始RCS × 距离衰减 × (复用次数衰减)
        matched_point = {
            'SR_RangeBin': sr_range_bin,
            'SR_AziBin': sr_azi_bin,
            'SR_EleBin': sr_ele_bin,
            'RCS': point_data['RCS'] * rcs_decay * reuse_factor,
            'doppler': point_data['doppler'],           # Doppler: 直接赋值
            'AbsV': point_data['AbsV'],
            'Vx': point_data['Vx'],
            'Vy': point_data['Vy'],
            'PowerBin': point_data['PowerBin'],
            'PhaseBin': point_data['PhaseBin'],
            'power': point_data['power'],
            'DopplerBin': point_data['DopplerBin'],
            'rcs_decay': rcs_decay,                    # 距离衰减因子（调试/可视化用）
            'reuse_count': reuse_n,                    # 复用次数（调试/可视化用）
            'reuse_db': reuse_db,                      # 复用次数衰减dB（调试/可视化用）
            'reuse_factor': reuse_factor,              # 复用次数线性衰减因子（调试/可视化用）
            'is_sr': is_sr_val,                        # 点来源：1=模型SR，0=AAI叠加
        }
        matched_points.append(matched_point)

    if max_match_per_orig > 0 and n_skipped_by_limit > 0:
        print(f"  RCS衰减: max L1_dist={max_l1_distance}, 跳过{n_skipped_by_limit}个SR点(原始点已达匹配上限{max_match_per_orig})")
    else:
        print(f"  RCS衰减: max L1_dist={max_l1_distance}, 无匹配次数限制")

    # 复用次数衰减统计
    if not enable_reuse_decay:
        n_reused = sum(reuse_hist.values())
        print(f"  复用次数衰减: 已禁用 (共{n_reused}个SR点复用同一原始点，未衰减RCS)")
    elif reuse_hist:
        parts = [f"N={n}: {reuse_hist[n]}pts({_reuse_count_to_db_attenuation(n):.0f}dB)"
                 for n in sorted(reuse_hist.keys())]
        n_reused = sum(reuse_hist.values())
        print(f"  复用次数衰减: 共{n_reused}个SR点复用被衰减 [{', '.join(parts)}]")
    else:
        print(f"  复用次数衰减: 无SR点复用同一原始点(N=1, 0dB)")

    return matched_points


def match_and_assign_values_neighborhood(
    sr_points: np.ndarray,
    original_data: np.ndarray,
    sr_rate1: int = 4,
    sr_rate2: int = 2,
    is_sr_arr: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    邻域填充模式（新路径，单方向线性插值，同Range相邻Azi→同Azi相邻Ele→同Ele相邻Range 回退）。

    原始点云投影到 SR 空间（origin_bin * sr_rate），SR 点沿单一方向在 base ±1 邻域
    [3 格]base-1, base, base+1内做线性插值：
    - 优先 同Range下相邻Azi；无邻居回退 同Azi下相邻Ele；再无回退 同Ele下相邻Range；三方向均无 → 丢弃
    - RCS = 方向内按 SR 子格位置 t∈[0,1) 线性插值（取 t 两侧最近两点；单点直接取值）
    - doppler / AbsV / Vx / Vy / PowerBin / PhaseBin / power / DopplerBin = 方向内距离最近点值
    - 若 SR 点与 base bin 原始点投影精确重合（三轴 t=0），所有字段直接取该点值
    - is_sr=0 的 AAI 叠加点走原 find_matches 最近点直接赋值
    """
    lookup_table = PointCloudLookupTable(original_data)
    range_dim = lookup_table.range_dim
    azi_dim = lookup_table.azi_dim
    ele_dim = lookup_table.ele_dim
    point_index_map = lookup_table.point_index_map

    # 预取字段数组（连续内存，便于批量 gather）
    rcs_arr = original_data['RCS'].astype(np.float32)
    doppler_arr = original_data['doppler'].astype(np.float32)
    abs_v_arr = original_data['AbsV'].astype(np.float32)
    vx_arr = original_data['Vx'].astype(np.float32)
    vy_arr = original_data['Vy'].astype(np.float32)
    power_arr = original_data['power'].astype(np.float32)
    power_bin_arr = original_data['PowerBin']
    phase_bin_arr = original_data['PhaseBin']
    doppler_bin_arr = original_data['DopplerBin']

    n_total = len(sr_points)
    if is_sr_arr is None:
        is_sr_arr = np.ones(n_total, dtype=np.uint8)
    is_sr_arr = np.asarray(is_sr_arr)

    sr_r_arr = np.asarray(sr_points[:, 0], dtype=np.int64)
    sr_a_arr = np.asarray(sr_points[:, 1], dtype=np.int64)
    sr_e_arr = np.asarray(sr_points[:, 2], dtype=np.int64)

    matched_points: List[Dict] = []
    n_discarded = 0
    n_direct_copy = 0
    n_coincidence = 0

    # ===== is_sr=0 分支：逐点 find_matches 直接赋值 =====
    is_sr0_mask = is_sr_arr == 0
    for i in np.flatnonzero(is_sr0_mask):
        sr_r = int(sr_r_arr[i]); sr_a = int(sr_a_arr[i]); sr_e = int(sr_e_arr[i])
        matched_indices = lookup_table.find_matches(sr_r, sr_a, sr_e, sr_rate1, sr_rate2)
        if len(matched_indices) == 0:
            n_discarded += 1
            continue
        point_data = lookup_table.get_point_data(matched_indices[0])
        matched_points.append({
            'SR_RangeBin': sr_r, 'SR_AziBin': sr_a, 'SR_EleBin': sr_e,
            'RCS': float(point_data['RCS']),
            'doppler': float(point_data['doppler']),
            'AbsV': float(point_data['AbsV']),
            'Vx': float(point_data['Vx']),
            'Vy': float(point_data['Vy']),
            'PowerBin': int(point_data['PowerBin']),
            'PhaseBin': int(point_data['PhaseBin']),
            'power': float(point_data['power']),
            'DopplerBin': int(point_data['DopplerBin']),
            'rcs_decay': 1.0, 'reuse_count': 1,
            'reuse_db': 0.0, 'reuse_factor': 1.0,
            'is_sr': 0,
        })
        n_direct_copy += 1

    # ===== is_sr=1 分支：单方向线性插值（Range→Ele→Azi 优先级回退） =====
    sr1_idx = np.flatnonzero(~is_sr0_mask)
    M = len(sr1_idx)
    if M > 0:
        sr_r1 = sr_r_arr[sr1_idx]
        sr_a1 = sr_a_arr[sr1_idx]
        sr_e1 = sr_e_arr[sr1_idx]
        base_r = sr_r1 // sr_rate1
        base_a = sr_a1 // sr_rate1
        base_e = sr_e1 // sr_rate2

        # 三方向子格位置 t ∈ [0, 1)
        t_R = (sr_r1 % sr_rate1) / sr_rate1
        t_A = (sr_a1 % sr_rate1) / sr_rate1
        t_E = (sr_e1 % sr_rate2) / sr_rate2

        base_r_safe = np.clip(base_r, 0, range_dim - 1)
        base_a_safe = np.clip(base_a, 0, azi_dim - 1)
        base_e_safe = np.clip(base_e, 0, ele_dim - 1)

        def _axis_interp(has_m1, has_0, has_p1, idx_m1, idx_0, idx_p1, t):
            """单方向RCS 线性插值 + 距离最近点索引。返回 (rcs[M], nearest_pi[M])。

            邻居位置 x ∈ {-1, 0, +1}，SR 子格位置 t ∈ [0,1)。
            取 t 两侧最近的两点做线性插值；单点直接取值。
            """
            m1_s = np.where(has_m1, idx_m1, 0)
            z_s = np.where(has_0, idx_0, 0)
            p1_s = np.where(has_p1, idx_p1, 0)
            R_m1 = rcs_arr[m1_s]; R_0 = rcs_arr[z_s]; R_p1 = rcs_arr[p1_s]
            # 6 种可用组合 (t∈[0,1] 时 x=0 始终在 t 左侧或重合，x=+1 始终在右侧)
            c1 = has_0 & has_p1                          # (0, +1)
            c2 = has_m1 & has_0 & ~has_p1               # (-1, 0)
            c3 = has_m1 & ~has_0 & has_p1               # (-1, +1)
            c4 = has_m1 & ~has_0 & ~has_p1              # only -1
            c5 = ~has_m1 & has_0 & ~has_p1              # only 0
            c6 = ~has_m1 & ~has_0 & has_p1              # only +1
            rcs = np.select(
                [c1, c2, c3, c4, c5, c6],
                [R_0 + t * (R_p1 - R_0),
                 R_m1 + (t + 1.0) * (R_0 - R_m1),
                 R_m1 + ((t + 1.0) / 2.0) * (R_p1 - R_m1),
                 R_m1, R_0, R_p1],
                default=0.0
            ).astype(np.float32)
            # 距离最近点：|0-t|, |+1-t|=1-t, |-1-t|=t+1
            low_t = t < 0.5
            near_low = np.select([has_0, has_m1, has_p1],
                                 [idx_0, idx_m1, idx_p1], default=-1)
            near_high = np.select([has_p1, has_0, has_m1],
                                  [idx_p1, idx_0, idx_m1], default=-1)
            nearest_pi = np.where(low_t, near_low, near_high)
            return rcs, nearest_pi

        # ----- Range 方向（变化 r，固定 a, e） -----
        r_m1 = base_r - 1; r_p1 = base_r + 1
        vm1 = (r_m1 >= 0) & (r_m1 < range_dim)
        vp1 = (r_p1 >= 0) & (r_p1 < range_dim)
        r_m1_s = np.clip(r_m1, 0, range_dim - 1)
        r_p1_s = np.clip(r_p1, 0, range_dim - 1)
        pi_rm1 = point_index_map[r_m1_s, base_a_safe, base_e_safe].astype(np.int64)
        pi_r0 = point_index_map[base_r_safe, base_a_safe, base_e_safe].astype(np.int64)
        pi_rp1 = point_index_map[r_p1_s, base_a_safe, base_e_safe].astype(np.int64)
        has_rm1 = vm1 & (pi_rm1 >= 0)
        has_r0 = (pi_r0 >= 0)
        has_rp1 = vp1 & (pi_rp1 >= 0)
        idx_rm1 = np.where(has_rm1, pi_rm1, -1)
        idx_r0 = np.where(has_r0, pi_r0, -1)
        idx_rp1 = np.where(has_rp1, pi_rp1, -1)
        rcs_R, near_R = _axis_interp(has_rm1, has_r0, has_rp1,
                                     idx_rm1, idx_r0, idx_rp1, t_R)
        has_any_R = has_rm1 | has_r0 | has_rp1

        # ----- Ele 方向（变化 e，固定 r, a） -----
        e_m1 = base_e - 1; e_p1 = base_e + 1
        vm1 = (e_m1 >= 0) & (e_m1 < ele_dim)
        vp1 = (e_p1 >= 0) & (e_p1 < ele_dim)
        e_m1_s = np.clip(e_m1, 0, ele_dim - 1)
        e_p1_s = np.clip(e_p1, 0, ele_dim - 1)
        pi_em1 = point_index_map[base_r_safe, base_a_safe, e_m1_s].astype(np.int64)
        pi_e0 = point_index_map[base_r_safe, base_a_safe, base_e_safe].astype(np.int64)
        pi_ep1 = point_index_map[base_r_safe, base_a_safe, e_p1_s].astype(np.int64)
        has_em1 = vm1 & (pi_em1 >= 0)
        has_e0 = (pi_e0 >= 0)
        has_ep1 = vp1 & (pi_ep1 >= 0)
        idx_em1 = np.where(has_em1, pi_em1, -1)
        idx_e0 = np.where(has_e0, pi_e0, -1)
        idx_ep1 = np.where(has_ep1, pi_ep1, -1)
        rcs_E, near_E = _axis_interp(has_em1, has_e0, has_ep1,
                                     idx_em1, idx_e0, idx_ep1, t_E)
        has_any_E = has_em1 | has_e0 | has_ep1

        # ----- Azi 方向（变化 a，固定 r, e） -----
        a_m1 = base_a - 1; a_p1 = base_a + 1
        vm1 = (a_m1 >= 0) & (a_m1 < azi_dim)
        vp1 = (a_p1 >= 0) & (a_p1 < azi_dim)
        a_m1_s = np.clip(a_m1, 0, azi_dim - 1)
        a_p1_s = np.clip(a_p1, 0, azi_dim - 1)
        pi_am1 = point_index_map[base_r_safe, a_m1_s, base_e_safe].astype(np.int64)
        pi_a0 = point_index_map[base_r_safe, base_a_safe, base_e_safe].astype(np.int64)
        pi_ap1 = point_index_map[base_r_safe, a_p1_s, base_e_safe].astype(np.int64)
        has_am1 = vm1 & (pi_am1 >= 0)
        has_a0 = (pi_a0 >= 0)
        has_ap1 = vp1 & (pi_ap1 >= 0)
        idx_am1 = np.where(has_am1, pi_am1, -1)
        idx_a0 = np.where(has_a0, pi_a0, -1)
        idx_ap1 = np.where(has_ap1, pi_ap1, -1)
        rcs_A, near_A = _axis_interp(has_am1, has_a0, has_ap1,
                                     idx_am1, idx_a0, idx_ap1, t_A)
        has_any_A = has_am1 | has_a0 | has_ap1

        # 精确重合：三轴 t=0 且 base bin 有原始点（base 在三方向中共享）
        at_base = (t_R == 0) & (t_A == 0) & (t_E == 0)
        coincidence_mask = at_base & has_r0
        n_coincidence = int(np.sum(coincidence_mask))

        # 优先级回退：同Range相邻Azi > 同Azi相邻Ele > 同Ele相邻Range；三方向均无 → 丢弃
        remaining = ~coincidence_mask
        azi_mask = remaining & has_any_A
        ele_mask = remaining & ~has_any_A & has_any_E
        range_mask = remaining & ~has_any_A & ~has_any_E & has_any_R
        discard_mask = remaining & ~has_any_A & ~has_any_E & ~has_any_R
        n_discarded += int(np.sum(discard_mask))

        # 按优先级合并 RCS 与 nearest_pi（低优先级先填，高优先级覆盖）
        final_rcs = np.zeros(M, dtype=np.float32)
        final_near = np.full(M, -1, dtype=np.int64)
        final_rcs = np.where(range_mask, rcs_R, final_rcs)
        final_near = np.where(range_mask, near_R, final_near)
        final_rcs = np.where(ele_mask, rcs_E, final_rcs)
        final_near = np.where(ele_mask, near_E, final_near)
        final_rcs = np.where(azi_mask, rcs_A, final_rcs)
        final_near = np.where(azi_mask, near_A, final_near)
        # 重合点：直接用 base 原始点
        base_pi = np.where(has_r0, pi_r0, 0)
        final_rcs = np.where(coincidence_mask, rcs_arr[base_pi], final_rcs)
        final_near = np.where(coincidence_mask, base_pi, final_near)

        # 从 nearest_pi 取其它字段
        near_safe = np.where(final_near >= 0, final_near, 0)
        doppler_v = doppler_arr[near_safe]
        abs_v_v = abs_v_arr[near_safe]
        vx_v = vx_arr[near_safe]; vy_v = vy_arr[near_safe]
        power_v = power_arr[near_safe]
        pbin_v = power_bin_arr[near_safe]
        phbin_v = phase_bin_arr[near_safe]
        dbin_v = doppler_bin_arr[near_safe]

        for j in range(M):
            if discard_mask[j]:
                continue
            matched_points.append({
                'SR_RangeBin': int(sr_r1[j]), 'SR_AziBin': int(sr_a1[j]),
                'SR_EleBin': int(sr_e1[j]),
                'RCS': float(final_rcs[j]),
                'doppler': float(doppler_v[j]),
                'AbsV': float(abs_v_v[j]),
                'Vx': float(vx_v[j]), 'Vy': float(vy_v[j]),
                'PowerBin': int(pbin_v[j]), 'PhaseBin': int(phbin_v[j]),
                'power': float(power_v[j]), 'DopplerBin': int(dbin_v[j]),
                'rcs_decay': 1.0, 'reuse_count': 1,
                'reuse_db': 0.0, 'reuse_factor': 1.0,
                'is_sr': 1,
            })

    n_interp = len(matched_points) - n_direct_copy - n_coincidence
    print(f"  邻域填充(单方向插值): 总SR点 {n_total}, 丢弃(三方向全空) {n_discarded}, "
          f"is_sr=0直接赋值 {n_direct_copy}, 精确重合 {n_coincidence}, 方向插值 {n_interp}")

    return matched_points


# ==================== PCD文件输出 ====================
def write_pcd_file(output_path: str, points: List[Dict], metadata: Dict,
                   add_is_sr: bool = False):
    """
    将超分点云写入PCD文件

    参数:
        output_path: 输出文件路径
        points: 超分点列表
        metadata: 元数据字典
        add_is_sr: 是否在输出PCD中追加 is_sr 字段（1=模型SR生成，0=AAI叠加的原始点云）。
                   默认False，输出与原格式一致。
    """
    num_points = len(points)
    if num_points == 0:
        print(f"警告：无有效点可写入 {output_path}")
        return

    # 基础字段列表（与原格式一致）
    base_fields = ['SR_x', 'SR_y', 'SR_z', 'SR_range', 'doppler', 'SR_sinAzi',
                   'SR_sinEle', 'RCS', 'power', 'AbsV', 'Vx', 'Vy',
                   'SR_RangeBin', 'DopplerBin', 'SR_AziBin', 'SR_EleBin',
                   'PowerBin', 'PhaseBin']

    # 按用户指定格式生成头部
    fields_line = ' '.join(base_fields)
    size_line = ' '.join(['4'] * 12 + ['2'] * 6)
    type_line = ' '.join(['F'] * 12 + ['I'] * 6)
    count_line = ' '.join(['1'] * 18)

    # 条件追加 is_sr 字段（与 RCS/Doppler 一致：4字节 float）
    extra_field_val = None
    if add_is_sr:
        fields_line += ' is_sr'
        size_line += ' 4'
        type_line += ' F'
        count_line += ' 1'

    header = (
        f"# metadata: RadarTimestamp={int(metadata.get('RadarTimestamp', 0))} "
        f"PacketTimeStamp={int(metadata.get('PacketTimeStamp', 0))} "
        f"VehicleSpeed={metadata.get('VehicleSpeed', 0):.5f} "
        f"YawRate={metadata.get('YawRate', 0):.6f}\n"
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {fields_line}\n"
        f"SIZE {size_line}\n"
        f"TYPE {type_line}\n"
        f"COUNT {count_line}\n"
        f"WIDTH {num_points}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {num_points}\n"
        "DATA binary\n"
    )

    # 定义输出数据类型（按FIELDS顺序）
    dtype_list = [
        ('SR_x', np.float32), ('SR_y', np.float32), ('SR_z', np.float32),
        ('SR_range', np.float32), ('doppler', np.float32), ('SR_sinAzi', np.float32),
        ('SR_sinEle', np.float32), ('RCS', np.float32), ('power', np.float32),
        ('AbsV', np.float32), ('Vx', np.float32), ('Vy', np.float32),
        ('SR_RangeBin', np.int16), ('DopplerBin', np.int16), ('SR_AziBin', np.int16),
        ('SR_EleBin', np.int16), ('PowerBin', np.int16), ('PhaseBin', np.int16)
    ]
    if add_is_sr:
        dtype_list.append(('is_sr', np.float32))
    dtype = np.dtype(dtype_list)

    # 创建输出数组
    output_data = np.zeros(num_points, dtype=dtype)

    for i, point in enumerate(points):
        base_vals = (
            point['SR_x'], point['SR_y'], point['SR_z'],
            point['SR_range'], point['doppler'], point['SR_sinAzi'],
            point['SR_sinEle'], point['RCS'], point['power'],
            point['AbsV'], point['Vx'], point['Vy'],
            point['SR_RangeBin'], point['DopplerBin'], point['SR_AziBin'],
            point['SR_EleBin'], point['PowerBin'], point['PhaseBin']
        )
        if add_is_sr:
            # 默认缺失时记为1.0（模型SR生成）
            is_sr_val = float(point.get('is_sr', 1))
            output_data[i] = base_vals + (is_sr_val,)
        else:
            output_data[i] = base_vals

    # 写入文件
    with open(output_path, 'wb') as f:
        f.write(header.encode('utf-8'))
        f.write(output_data.tobytes())


# ==================== 主处理函数 ====================
def process_pcd_file(input_path: str, output_path: Optional[str], model: nn.Module,
                     device: torch.device, model_type: str = "orin",
                     threshold: float = 0.5, sr_rate1: int = 4, sr_rate2: int = 2,
                     use_amp: bool = False,
                     enable_visualization: bool = False,
                     vis_save_dir: Optional[str] = None,
                     occ_gt_path: Optional[str] = None,
                     eval_results: Optional[List[Dict]] = None,
                     use_aai: bool = False,
                     use_original_overlay: bool = False,
                     use_dynamic_static: bool = True,
                     add_offset: bool = True,
                     max_match_per_orig: int = 0,
                     enable_reuse_decay: bool = True,
                     add_is_sr: bool = False,
                     use_neighborhood_filling: bool = False,
                     preserve_original_points: bool = False,
                     sr_min_rcs: Optional[float] = None,
                     sr_min_abs_v: Optional[float] = None,
                     sr_static_min_rcs: Optional[float] = None,
                     sr_min_range: Optional[float] = None,
                     sr_max_range: Optional[float] = None,
                     sr_empty_voxel_size: Optional[Tuple[float, float]] = None,
                     expand_dynamic_raw: bool = False,
                     raw_expand_min_abs_v: float = 1.5,
                     raw_expand_min_rcs: float = 10.0,
                     raw_expand_max_range: float = 50.0,
                     raw_expand_voxel_size: Tuple[float, float] = (0.25, 0.20),
                     raw_expand_rcs_scale: float = 1.0,
                     raw_expand_absv_scale: float = 1.0,
                     expand_dense_raw: bool = False,
                     dense_expand_min_points: int = 8,
                     dense_expand_min_rcs: float = 5.0,
                     dense_expand_max_abs_v: float = 0.5,
                     dense_expand_max_range: float = 50.0,
                     dense_expand_adaptive_axis: bool = False,
                     dense_expand_axis_radius: float = 3.0,
                     dense_expand_min_axis_ratio: float = 1.0,
                     dense_expand_lateral_steps: int = 1,
                     dense_expand_lateral_min_ratio: float = 1.0,
                     dense_expand_keep_longitudinal: bool = False,
                     dense_expand_require_adaptive_axis: bool = False,
                     bridge_dense_raw: bool = False,
                     dense_bridge_max_gap: float = 1.5,
                     dense_bridge_min_axis_ratio: float = 10.0):
    """
    处理单个PCD文件的超分辨率流程

    参数:
        input_path: 输入PCD文件路径
        output_path: 输出PCD文件路径（可选，None则不保存输出文件）
        model: U-Net模型
        device: 计算设备
        model_type: 模型类型 ("orin" 或 "zynq")
        threshold: 输出阈值
        sr_rate1: Range/Azi超分倍数
        sr_rate2: Ele超分倍数
        use_amp: 是否使用混合精度
        enable_visualization: 是否启用点云可视化
        vis_save_dir: 可视化图片保存目录（可选，None则交互式显示）
        occ_gt_path: Occ真值文件路径（coord格式，用于评估）
        eval_results: 评估结果列表（用于累积评估结果）
        use_aai: 是否使用AAI精度提升叠加
        use_original_overlay: 是否直接叠加原始点云（跳过AAI处理），将原始雷达点云
                              缩放到SR分辨率后直接与SR点云合并。use_aai优先级高于此参数。
        use_dynamic_static: 是否使用动静态分离的AAI处理（与aai_test.py 一致），默认True
        add_offset: 是否对模型输入的 RCS 和 Doppler 加70偏置（与训练一致），默认True。
                    ORIN 密集模型始终不加偏置（其输入不需要）。
        max_match_per_orig: 每个原始点最大被SR点匹配次数。
                            >0 时启用限制；0 表示不限制（默认）。
                            RCS 采用 bin 距离线性衰减，Doppler 直接赋值。
        enable_reuse_decay: 是否启用复用次数衰减（能量均分），默认True。
                            False时RCS仅受bin距离线性衰减，不复用衰减。
        add_is_sr: 是否在输出PCD中追加 is_sr 字段（1=模型SR生成，0=AAI叠加的原始点云）。
                   默认False，输出与原格式一致。
        use_neighborhood_filling: 是否启用邻域填充模式（新路径，默认False）。
                                  True时RCS用3轴邻域count/加权平均，doppler/AbsV用中位数代表法；
                                  从邻域选值，3轴全空丢弃SR点，is_sr=0点仍走最近点直接赋值。
                                  False时走原 match_and_assign_values 路径。
        preserve_original_points: 将输入PCD中的原始x/y/z和五维特征逐点保留到输出，
                                  避免把原始bin重建坐标替换掉原始测量。
        sr_min_rcs: 仅保留RCS不低于该阈值的模型生成点；None表示不按RCS筛选。
        sr_min_abs_v: 仅保留绝对速度不低于该阈值的模型生成点；None表示不筛选。
        sr_static_min_rcs: 与sr_min_abs_v组合为OR门控；低速点达到此RCS仍可保留。
        sr_min_range/sr_max_range: 模型生成点的物理距离门控（米）。
        sr_empty_voxel_size: (x,y) 网格尺寸。设置后仅向原始点未占用的网格填充，
                             每个空网格只保留RCS最高的一个SR点。
        expand_dynamic_raw: 将近距高RCS动态原始回波沿纵向扩展到相邻空网格。
        raw_expand_*: 动态原始回波扩展的特征门控、距离和网格尺寸。
        raw_expand_rcs_scale/raw_expand_absv_scale: 合成原始支持点的 RCS/AbsV
            缩放；原始测量点保持精确不变。
        expand_dense_raw: 将近距慢速、同一检测网格内多回波的原始点沿纵向扩展。
        dense_expand_*: 密集慢速回波的点数、RCS、速度和距离门控。
        dense_expand_adaptive_axis: 使用邻近合格网格的PCA主轴选择纵向或横向扩展。
        dense_expand_min_axis_ratio: 仅当PCA主/次特征值比达到阈值时改为横向扩展。
        dense_expand_lateral_steps: 横向扩展的半径（网格步数）。
        dense_expand_lateral_min_ratio: 只有PCA各向异性达到该阈值时才使用额外横向步数。
        dense_expand_keep_longitudinal: 自适应横向扩展时同时保留原纵向邻格。
        dense_expand_require_adaptive_axis: 丢弃未通过PCA横向门控的普通密集慢速种子。
        bridge_dense_raw: 在高各向异性横向慢速密集种子之间插值内部空网格。
    """
    # 1. 解析PCD文件
    metadata, data, header_lines = parse_pcd_file(input_path)
    print(f"  加载点数: {len(data)}")

    if len(data) == 0:
        print(f"  文件无点云数据，跳过")
        return

    # 2. 创建输入张量（与 EnhancedBinDataset 格式一致）
    # ORIN 密集模型输入不需要偏置；ZYNQ 稀疏模型由 add_offset 参数控制
    effective_add_offset = add_offset and model_type != "orin"
    sparse_tensor = create_sparse_input(data, device=device, add_offset=effective_add_offset, metadata=metadata)

    # 3. 根据模型类型准备输入
    if model_type == "orin":
        # ORIN 模型使用密集张量 [1, 64, 512, 128]
        input_tensor = sparse_tensor.dense()
    else:
        # ZYNQ 模型使用稀疏张量
        input_tensor = sparse_tensor

    # 4. 执行推理，获取点云坐标 (RangeBin, AziBin, EleBin)
    points = run_inference(
        model, input_tensor, device,
        threshold=threshold,
        model_type=model_type,
        use_amp=use_amp
    )
    # points 格式: [N_points, 3] (RangeBin, AziBin, EleBin)

    print(f"  模型输出: {len(points)} 个超分点 (阈值={threshold})")

    if len(points) == 0:
        print(f"  无超分点提取，跳过")
        return

    # 5. 转换为 numpy 数组
    points_np = points.cpu().numpy()  # [N_points, 3]

    # 6. 原始点云叠加处理（AAI处理或直接叠加）
    aai_xyz = None
    aai_data = None
    aai_dynamic_mask = None
    original_dynamic_mask = None
    merged_points_np = points_np  # 默认使用超分点云
    merged_is_sr = np.ones(len(points_np), dtype=np.uint8) if len(points_np) > 0 \
        else np.empty((0,), dtype=np.uint8)  # 默认全部为模型SR生成
    merge_stats = None

    if use_aai or use_original_overlay:
        # 从原始数据提取 bin 坐标
        # 将结构化数组转换为 [N, 6] 格式
        raw_data_for_aai = np.zeros((len(data), 6), dtype=np.float32)
        raw_data_for_aai[:, COL_RANGE] = data['RangeBin'].astype(np.float32)
        raw_data_for_aai[:, COL_AZI] = data['AziBin'].astype(np.float32)
        raw_data_for_aai[:, COL_ELE] = data['EleBin'].astype(np.float32)
        raw_data_for_aai[:, COL_DOPPLER] = data['doppler'].astype(np.float32)
        raw_data_for_aai[:, COL_RCS] = data['RCS'].astype(np.float32)
        raw_data_for_aai[:, COL_PHASE] = data['PhaseBin'].astype(np.float32)

        if use_aai:
            # === AAI处理路径 ===
            overlay_label = "AAI"
            # 执行AAI处理（与 aai_test.py 一致）
            if use_dynamic_static:
                x_aai, y_aai, z_aai, df_final, point_validity = aai_frame3_dynamic_static(raw_data_for_aai, metadata)
                # 原始点云的动静态标记（point_validity 1:1 对应输入 raw_data_for_aai）
                original_dynamic_mask = (point_validity == 8)
            else:
                x_aai, y_aai, z_aai, df_final = aai_frame3(raw_data_for_aai)

            # 转换为标准的 numpy 格式
            if len(x_aai) > 0:
                aai_xyz = np.stack([x_aai, y_aai, z_aai], axis=1).astype(np.float32)
                aai_data = df_final[COL_NAMES].values.astype(np.float32)
                # 提取动态/静态标记（仅动静态分离模式有效）
                if use_dynamic_static and 'IsDynamic' in df_final.columns:
                    aai_dynamic_mask = df_final['IsDynamic'].values.astype(bool)
            else:
                aai_xyz = np.empty((0, 3), dtype=np.float32)
                aai_data = np.empty((0, 6), dtype=np.float32)
            print(f"  AAI处理后: {len(aai_xyz)} 个点 (原始: {len(data)})")
            overlay_data = aai_data
        else:
            # === 直接叠加原始点云路径（跳过AAI处理） ===
            overlay_label = "Original"
            overlay_data = raw_data_for_aai
            # 计算原始点云xyz坐标（用于可视化）
            ori_x, ori_y, ori_z = calculate_original_coordinates(data)
            aai_xyz = np.stack([ori_x, ori_y, ori_z], axis=1).astype(np.float32)
            aai_dynamic_mask = None
            original_dynamic_mask = None
            print(f"  直接叠加原始点云: {len(aai_xyz)} 个点 (未经AAI处理)")

        if overlay_data.shape[0] > 0:
            # 在bin层级叠加SR和原始点云，同时获得每个点的来源标记
            merged_points_np, merge_stats, merged_is_sr = merge_sr_with_aai_bins(
                points_np, overlay_data, sr_rate1, sr_rate2
            )
            print(f"  {overlay_label}叠加后: {len(merged_points_np)} 个bin点 "
                  f"(SR生成: {int(np.sum(merged_is_sr))}, 叠加: {int(np.sum(merged_is_sr == 0))})")
            if merge_stats:
                print(f"    SR独有: {merge_stats['sr_only_bins']}, 新增: {merge_stats['aai_only_bins']}, 重叠: {merge_stats['overlap_bins']}")

    # 7. 使用叠加后的bin数据进行匹配和赋值
    if use_neighborhood_filling:
        matched_points = match_and_assign_values_neighborhood(
            merged_points_np, data, sr_rate1, sr_rate2,
            is_sr_arr=merged_is_sr)
    else:
        matched_points = match_and_assign_values(merged_points_np, data, sr_rate1, sr_rate2,
                                                 max_match_per_orig=max_match_per_orig,
                                                 is_sr_arr=merged_is_sr,
                                                 enable_reuse_decay=enable_reuse_decay)
    print(f"  匹配点数: {len(matched_points)}")

    if len(matched_points) == 0:
        print(f"  无匹配点，跳过")
        return

    # 8. 计算坐标并过滤无效点（批量处理）
    # 提取所有匹配点的bin值
    sr_range_bins = np.array([p['SR_RangeBin'] for p in matched_points], dtype=np.float32)
    sr_azi_bins = np.array([p['SR_AziBin'] for p in matched_points], dtype=np.float32)
    sr_ele_bins = np.array([p['SR_EleBin'] for p in matched_points], dtype=np.float32)

    # 批量计算坐标
    SR_x, SR_y, SR_z, SR_range, SR_sinAzi, SR_sinEle, valid_geom_mask = \
        calculate_sr_coordinates_batched(sr_range_bins, sr_azi_bins, sr_ele_bins, sr_rate1, sr_rate2)

    # 批量FOV检查
    fov_valid_mask = is_valid_fov_batched(SR_sinAzi, SR_sinEle)

    # 综合过滤掩码
    final_valid_mask = valid_geom_mask & fov_valid_mask

    # 批量添加坐标并筛选有效点
    valid_points = []
    for i, point in enumerate(matched_points):
        if not final_valid_mask[i]:
            continue
        point['SR_x'] = SR_x[i]
        point['SR_y'] = SR_y[i]
        point['SR_z'] = SR_z[i]
        point['SR_range'] = SR_range[i]
        point['SR_sinAzi'] = SR_sinAzi[i]
        point['SR_sinEle'] = SR_sinEle[i]
        valid_points.append(point)

    print(f"  FOV过滤后有效点: {len(valid_points)}")

    # 保守策略：保留原始测量的精确坐标/特征，再加入经过匹配的模型点。
    # 原始点不再经过 bin->xyz 重建，避免重建误差破坏 raw 分支已有性能。
    if preserve_original_points:
        raw_bin_set = {
            (int(data['RangeBin'][i]) * sr_rate1,
             int(data['AziBin'][i]) * sr_rate1,
             int(data['EleBin'][i]) * sr_rate2)
            for i in range(len(data))
        }
        sr_candidates = []
        n_duplicate = 0
        n_rcs_rejected = 0
        n_absv_rejected = 0
        n_range_rejected = 0
        for point in valid_points:
            if int(point.get('is_sr', 1)) == 0:
                continue
            point_bin = (int(point['SR_RangeBin']), int(point['SR_AziBin']),
                         int(point['SR_EleBin']))
            if point_bin in raw_bin_set:
                n_duplicate += 1
                continue
            if sr_min_rcs is not None and float(point.get('RCS', -np.inf)) < sr_min_rcs:
                n_rcs_rejected += 1
                continue
            if sr_min_abs_v is not None:
                is_dynamic = abs(float(point.get('AbsV', 0.0))) >= sr_min_abs_v
                is_strong_static = (sr_static_min_rcs is not None and
                                    float(point.get('RCS', -np.inf)) >= sr_static_min_rcs)
                if not is_dynamic and not is_strong_static:
                    n_absv_rejected += 1
                    continue
            point_range = float(point.get('SR_range', np.inf))
            if ((sr_min_range is not None and point_range < sr_min_range) or
                    (sr_max_range is not None and point_range >= sr_max_range)):
                n_range_rejected += 1
                continue
            sr_candidates.append(point)

        n_occupied_voxel = 0
        n_voxel_dedup = 0
        if sr_empty_voxel_size is not None:
            voxel_x, voxel_y = map(float, sr_empty_voxel_size)
            if voxel_x <= 0 or voxel_y <= 0:
                raise ValueError('sr_empty_voxel_size values must be positive')
            raw_voxels = {
                (math.floor(float(data['x'][i]) / voxel_x),
                 math.floor(float(data['y'][i]) / voxel_y))
                for i in range(len(data))
            }
            best_by_voxel = {}
            for point in sr_candidates:
                key = (math.floor(float(point['SR_x']) / voxel_x),
                       math.floor(float(point['SR_y']) / voxel_y))
                if key in raw_voxels:
                    n_occupied_voxel += 1
                    continue
                old = best_by_voxel.get(key)
                if old is None or float(point['RCS']) > float(old['RCS']):
                    if old is not None:
                        n_voxel_dedup += 1
                    best_by_voxel[key] = point
                else:
                    n_voxel_dedup += 1
            sr_kept = list(best_by_voxel.values())
        else:
            sr_kept = sr_candidates

        def _raw_field(name, i, default=0.0):
            return float(data[name][i]) if name in data.dtype.names else float(default)

        original_points = []
        for i in range(len(data)):
            original_points.append({
                'SR_x': _raw_field('x', i), 'SR_y': _raw_field('y', i),
                'SR_z': _raw_field('z', i), 'SR_range': _raw_field('range', i),
                'SR_sinAzi': _raw_field('sinAzi', i), 'SR_sinEle': _raw_field('sinEle', i),
                'RCS': _raw_field('RCS', i), 'doppler': _raw_field('doppler', i),
                'AbsV': _raw_field('AbsV', i), 'Vx': _raw_field('Vx', i),
                'Vy': _raw_field('Vy', i), 'power': _raw_field('power', i),
                'SR_RangeBin': int(data['RangeBin'][i]) * sr_rate1,
                'DopplerBin': int(data['DopplerBin'][i]),
                'SR_AziBin': int(data['AziBin'][i]) * sr_rate1,
                'SR_EleBin': int(data['EleBin'][i]) * sr_rate2,
                'PowerBin': int(data['PowerBin'][i]), 'PhaseBin': int(data['PhaseBin'][i]),
                'rcs_decay': 1.0, 'reuse_count': 1, 'reuse_db': 0.0,
                'reuse_factor': 1.0, 'is_sr': 0,
            })

        expanded_points = []
        n_dynamic_expanded = 0
        n_dense_expanded = 0
        n_bridge_expanded = 0
        if expand_dynamic_raw or expand_dense_raw:
            voxel_x, voxel_y = map(float, raw_expand_voxel_size)
            if voxel_x <= 0 or voxel_y <= 0:
                raise ValueError('raw_expand_voxel_size values must be positive')
            raw_indices_by_voxel = {}
            for i in range(len(data)):
                raw_key = (math.floor(float(data['x'][i]) / voxel_x),
                           math.floor(float(data['y'][i]) / voxel_y))
                raw_indices_by_voxel.setdefault(raw_key, []).append(i)
            raw_xy_voxels = set(raw_indices_by_voxel)
            best_seed_by_target = {}

            def _propose_offsets(source, seed_i, offsets):
                for offset_x, offset_y in offsets:
                    target = (source[0] + offset_x, source[1] + offset_y)
                    if target in raw_xy_voxels:
                        continue
                    old_i = best_seed_by_target.get(target)
                    if old_i is None or _raw_field('RCS', seed_i) > _raw_field('RCS', old_i):
                        best_seed_by_target[target] = seed_i

            if expand_dynamic_raw:
                for i in range(len(data)):
                    raw_range = _raw_field('range', i, np.linalg.norm([
                        _raw_field('x', i), _raw_field('y', i), _raw_field('z', i)]))
                    if (raw_range >= raw_expand_max_range or
                            abs(_raw_field('AbsV', i)) < raw_expand_min_abs_v or
                            _raw_field('RCS', i) < raw_expand_min_rcs):
                        continue
                    source = (math.floor(_raw_field('x', i) / voxel_x),
                              math.floor(_raw_field('y', i) / voxel_y))
                    _propose_offsets(source, i, ((-1, 0), (1, 0)))
                n_dynamic_expanded = len(best_seed_by_target)

            if expand_dense_raw:
                if dense_expand_min_points < 1:
                    raise ValueError('dense_expand_min_points must be at least 1')
                if dense_expand_axis_radius <= 0:
                    raise ValueError('dense_expand_axis_radius must be positive')
                if dense_expand_lateral_steps < 1:
                    raise ValueError('dense_expand_lateral_steps must be at least 1')
                if dense_expand_lateral_min_ratio <= 0:
                    raise ValueError('dense_expand_lateral_min_ratio must be positive')
                dense_seeds = []
                for source, voxel_indices in raw_indices_by_voxel.items():
                    if len(voxel_indices) < dense_expand_min_points:
                        continue
                    seed_i = max(voxel_indices, key=lambda idx: _raw_field('RCS', idx))
                    raw_range = _raw_field('range', seed_i, np.linalg.norm([
                        _raw_field('x', seed_i), _raw_field('y', seed_i),
                        _raw_field('z', seed_i)]))
                    if (raw_range >= dense_expand_max_range or
                            _raw_field('RCS', seed_i) < dense_expand_min_rcs or
                            abs(_raw_field('AbsV', seed_i)) >= dense_expand_max_abs_v):
                        continue
                    dense_seeds.append((source, seed_i))

                dense_centers = np.asarray([
                    ((source[0] + 0.5) * voxel_x, (source[1] + 0.5) * voxel_y)
                    for source, _ in dense_seeds
                ], dtype=np.float32)
                for seed_index, (source, seed_i) in enumerate(dense_seeds):
                    offsets = ((-1, 0), (1, 0))
                    adaptive_lateral = False
                    axis_ratio = 0.0
                    if dense_expand_adaptive_axis and len(dense_centers) >= 2:
                        delta = dense_centers - dense_centers[seed_index]
                        nearby = dense_centers[
                            np.sum(delta * delta, axis=1) <= dense_expand_axis_radius ** 2
                        ]
                        if len(nearby) >= 2:
                            centered = nearby - np.mean(nearby, axis=0, keepdims=True)
                            covariance = centered.T @ centered / float(len(nearby))
                            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                            major_axis = eigenvectors[:, -1]
                            axis_ratio = float(eigenvalues[-1]) / max(float(eigenvalues[0]), 1e-9)
                            if (axis_ratio >= dense_expand_min_axis_ratio and
                                    abs(float(major_axis[1])) > abs(float(major_axis[0]))):
                                adaptive_lateral = True
                                if dense_expand_keep_longitudinal:
                                    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
                                else:
                                    offsets = ((0, -1), (0, 1))
                                if dense_expand_lateral_steps > 1 and axis_ratio >= dense_expand_lateral_min_ratio:
                                    lateral_offsets = []
                                    for step in range(1, int(dense_expand_lateral_steps) + 1):
                                        lateral_offsets.extend(((0, -step), (0, step)))
                                    offsets = tuple(offsets) + tuple(lateral_offsets)
                    if dense_expand_require_adaptive_axis and not adaptive_lateral:
                        continue
                    _propose_offsets(source, seed_i, offsets)
                n_dense_expanded = len(best_seed_by_target) - n_dynamic_expanded

                if bridge_dense_raw and len(dense_centers) >= 2:
                    if dense_bridge_max_gap <= 0:
                        raise ValueError('dense_bridge_max_gap must be positive')
                    before_bridge = len(best_seed_by_target)
                    for seed_index, (source, seed_i) in enumerate(dense_seeds):
                        delta = dense_centers - dense_centers[seed_index]
                        distances = np.sqrt(np.sum(delta * delta, axis=1))
                        local = dense_centers[distances <= dense_expand_axis_radius]
                        if len(local) < 2:
                            continue
                        centered = local - np.mean(local, axis=0, keepdims=True)
                        eigenvalues, eigenvectors = np.linalg.eigh(
                            centered.T @ centered / float(len(local)))
                        ratio = float(eigenvalues[-1]) / max(float(eigenvalues[0]), 1e-9)
                        major_axis = eigenvectors[:, -1]
                        if (ratio < dense_bridge_min_axis_ratio or
                                abs(float(major_axis[1])) <= abs(float(major_axis[0]))):
                            continue
                        neighbors = np.flatnonzero(
                            (distances > 0) &
                            (distances <= dense_bridge_max_gap) &
                            (np.abs(delta[:, 1]) > np.abs(delta[:, 0]))
                        )
                        if len(neighbors) == 0:
                            continue
                        neighbor_index = int(neighbors[np.argmin(distances[neighbors])])
                        neighbor_source = dense_seeds[neighbor_index][0]
                        steps = max(abs(neighbor_source[0] - source[0]),
                                    abs(neighbor_source[1] - source[1]))
                        for step in range(1, steps):
                            fraction = step / float(steps)
                            target = (
                                int(round(source[0] + fraction * (neighbor_source[0] - source[0]))),
                                int(round(source[1] + fraction * (neighbor_source[1] - source[1]))),
                            )
                            if target in raw_xy_voxels:
                                continue
                            old_i = best_seed_by_target.get(target)
                            if (old_i is None or
                                    _raw_field('RCS', seed_i) > _raw_field('RCS', old_i)):
                                best_seed_by_target[target] = seed_i
                    n_bridge_expanded = len(best_seed_by_target) - before_bridge

            if raw_expand_rcs_scale < 0 or raw_expand_absv_scale < 0:
                raise ValueError('raw expansion feature scales must be non-negative')
            for target, i in best_seed_by_target.items():
                point = original_points[i].copy()
                point['SR_x'] = (target[0] + 0.5) * voxel_x
                point['SR_y'] = (target[1] + 0.5) * voxel_y
                point['SR_range'] = float(math.sqrt(
                    point['SR_x'] ** 2 + point['SR_y'] ** 2 + point['SR_z'] ** 2))
                point['RCS'] *= float(raw_expand_rcs_scale)
                point['AbsV'] *= float(raw_expand_absv_scale)
                point['is_sr'] = 1
                expanded_points.append(point)

            # Merge learned-SR and deterministic support expansion without
            # writing more than one generated point to the same detection cell.
            best_generated = {}
            for point in sr_kept + expanded_points:
                key = (math.floor(float(point['SR_x']) / voxel_x),
                       math.floor(float(point['SR_y']) / voxel_y))
                old = best_generated.get(key)
                if old is None or float(point['RCS']) > float(old['RCS']):
                    best_generated[key] = point
            sr_kept = list(best_generated.values())
        valid_points = original_points + sr_kept
        print(f"  保留原始点精确坐标: {len(original_points)}; 保留SR点: {len(sr_kept)}; "
              f"重合bin丢弃: {n_duplicate}; RCS阈值丢弃: {n_rcs_rejected}; "
              f"AbsV阈值丢弃: {n_absv_rejected}; 已占用voxel丢弃: {n_occupied_voxel}; "
              f"距离门控丢弃: {n_range_rejected}; "
              f"voxel内去重: {n_voxel_dedup}; "
              f"动态原始扩展候选: {n_dynamic_expanded}; "
              f"密集慢速扩展新增候选: {n_dense_expanded}; "
              f"密集簇内部桥接新增候选: {n_bridge_expanded}; "
              f"原始扩展合计: {len(expanded_points)}; "
              f"最终: {len(valid_points)}")

    # 9. Occ真值评估（可选）
    if occ_gt_path and eval_results is not None:
        occ_data = load_occ_gt_coord(occ_gt_path)
        if occ_data is not None:
            # 提取超分点云xyz坐标
            sr_xyz = np.array([[p['SR_x'], p['SR_y'], p['SR_z']]
                               for p in valid_points], dtype=np.float32)
            result = evaluate_frame(sr_xyz, occ_data)
            result['frame_name'] = os.path.basename(input_path)
            result['use_aai'] = use_aai
            if merge_stats:
                result['merge_stats'] = merge_stats
            eval_results.append(result)
            print(f"  评估: ACC={result['acc']:.4f}, IoU={result['iou']:.4f}, PPV={result['ppv']:.4f}")

    # 10. 可视化（可选）
    if enable_visualization:
        # 获取文件名作为标题
        file_name = os.path.basename(input_path)
        vis_title = f"Point Cloud: {file_name}"

        # 确定保存路径
        if vis_save_dir:
            os.makedirs(vis_save_dir, exist_ok=True)
            base_name = os.path.splitext(file_name)[0]
            vis_save_path = os.path.join(vis_save_dir, f"{base_name}_vis.png")
            doppler_vis_save_path = os.path.join(vis_save_dir, f"{base_name}_doppler_vis.png")
            aai_vis_save_path = os.path.join(vis_save_dir, f"{base_name}_aai_comparison.png")
        else:
            vis_save_path = None
            doppler_vis_save_path = None
            aai_vis_save_path = None

        # 调用可视化函数（包含 RCS 颜色编码）
        visualize_point_clouds(
            original_data=data,
            sr_points_np=points_np,
            valid_points=valid_points,
            sr_rate1=sr_rate1,
            sr_rate2=sr_rate2,
            title=vis_title,
            save_path=vis_save_path
        )

        # 调用 Doppler 专用可视化函数
        visualize_point_clouds_doppler(
            original_data=data,
            sr_points_np=points_np,
            valid_points=valid_points,
            sr_rate1=sr_rate1,
            sr_rate2=sr_rate2,
            title=f"Doppler Analysis: {file_name}",
            save_path=doppler_vis_save_path
        )

        # 叠加对比可视化（如果启用AAI或直接叠加）
        if (use_aai or use_original_overlay) and aai_xyz is not None and len(aai_xyz) > 0:
            # 计算原始点云的xyz坐标（用于可视化）
            ori_x, ori_y, ori_z = calculate_original_coordinates(data)
            original_xyz_vis = np.stack([ori_x, ori_y, ori_z], axis=1).astype(np.float32)

            # 计算SR点云的xyz坐标（用于可视化）
            sr_range_bins_vis = points_np[:, 0].astype(np.float32)
            sr_azi_bins_vis = points_np[:, 1].astype(np.float32)
            sr_ele_bins_vis = points_np[:, 2].astype(np.float32)
            SR_x_vis, SR_y_vis, SR_z_vis, _, _, _, valid_mask_vis = \
                calculate_sr_coordinates_batched(sr_range_bins_vis, sr_azi_bins_vis, sr_ele_bins_vis, sr_rate1, sr_rate2)
            sr_xyz_vis = np.stack([SR_x_vis, SR_y_vis, SR_z_vis], axis=1)
            sr_xyz_vis = sr_xyz_vis[valid_mask_vis]

            # 计算合并后的xyz坐标
            merged_range_bins = merged_points_np[:, 0].astype(np.float32)
            merged_azi_bins = merged_points_np[:, 1].astype(np.float32)
            merged_ele_bins = merged_points_np[:, 2].astype(np.float32)
            merged_x, merged_y, merged_z, _, _, _, merged_valid_mask = \
                calculate_sr_coordinates_batched(merged_range_bins, merged_azi_bins, merged_ele_bins, sr_rate1, sr_rate2)
            merged_xyz_vis = np.stack([merged_x, merged_y, merged_z], axis=1)
            merged_xyz_vis = merged_xyz_vis[merged_valid_mask]

            # 加载Occ真值（如果有）
            occ_gt_xyz = None
            if occ_gt_path:
                occ_data_vis = load_occ_gt_coord(occ_gt_path)
                if occ_data_vis is not None and occ_data_vis.shape[0] > 0:
                    # 只取有效标签（2或3）
                    valid_occ_mask = (occ_data_vis[:, 3] == 2) | (occ_data_vis[:, 3] == 3)
                    valid_occ = occ_data_vis[valid_occ_mask]
                    # 从网格索引转换为xyz坐标（使用网格中心）
                    # GridY从1开始，GridX/GridZ有正负（>=0从1开始，<0从-1开始）
                    if valid_occ.shape[0] > 0:
                        occ_gt_xyz = np.zeros((valid_occ.shape[0], 3), dtype=np.float32)

                        grid_y = valid_occ[:, 0].astype(np.float32)
                        grid_x = valid_occ[:, 1].astype(np.float32)
                        grid_z = valid_occ[:, 2].astype(np.float32)

                        # GridY: 从1开始, y = (GridY-1)*res + res/2
                        occ_gt_xyz[:, 1] = (grid_y - 1) * GRID_RES + GRID_RES / 2

                        # GridX: 正值从1开始，负值从-1开始
                        # x >= 0: GridX = int(x/res) + 1, 所以 x = (GridX-1)*res + res/2
                        # x < 0: GridX = int(x/res) - 1, 所以 x = (GridX+1)*res + res/2
                        pos_mask_x = grid_x >= 0
                        occ_gt_xyz[:, 0] = np.where(pos_mask_x,
                                                    (grid_x - 1) * GRID_RES + GRID_RES / 2,
                                                    (grid_x + 1) * GRID_RES + GRID_RES / 2)

                        # GridZ: 同GridX逻辑
                        pos_mask_z = grid_z >= 0
                        occ_gt_xyz[:, 2] = np.where(pos_mask_z,
                                                    (grid_z - 1) * GRID_RES + GRID_RES / 2,
                                                    (grid_z + 1) * GRID_RES + GRID_RES / 2)

            visualize_aai_comparison(
                sr_xyz=sr_xyz_vis,
                aai_xyz=aai_xyz,
                merged_xyz=merged_xyz_vis,
                occ_gt_xyz=occ_gt_xyz,
                grid_res=GRID_RES,
                title=f"AAI Comparison: {file_name}",
                save_path=aai_vis_save_path,
                merge_stats=merge_stats,
                aai_dynamic_mask=aai_dynamic_mask,
                original_xyz=original_xyz_vis,
                original_dynamic_mask=original_dynamic_mask
            )

    # 11. 写入输出PCD文件（仅当指定了输出路径时）
    if output_path is not None:
        write_pcd_file(output_path, valid_points, metadata, add_is_sr=add_is_sr)
        print(f"  已保存: {output_path}")


def _parse_bool_arg(value):
    """Parse booleans reliably (``bool('False')`` is incorrectly True)."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无效布尔值: {value!r}")


def discover_pcd_files(input_dir: str, recursive: bool = False) -> List[Path]:
    """Discover direct PCD files or all nested radar_front_bottom files."""
    root = Path(input_dir).expanduser().resolve()
    direct = sorted(root.glob("*.pcd"))
    if direct and not recursive:
        return direct

    nested = sorted(root.glob("**/radar_front_bottom/*.pcd"))
    if nested:
        return nested
    return direct


def main():
    """
    主函数：批量处理PCD文件
    支持命令行参数指定文件索引范围和AAI叠加功能
    """
    import argparse

    parser = argparse.ArgumentParser(description='雷达点云超分辨率推理（支持AAI角精度提升叠加）')
    parser.add_argument('--input_dir', type=str, default="./frame200_ori",
                        help='输入PCD目录；传入frame200_ori根目录时可递归扫描所有radar_front_bottom')
    parser.add_argument('--recursive', action='store_true',
                        help='递归查找所有 **/radar_front_bottom/*.pcd')
    parser.add_argument('--output_dir', type=str, default="./output/radar_front_bottom_sr",
                        help='输出PCD文件目录')
    parser.add_argument('--model_path', type=str, default="./model_epoch_60.pth",
                        help='模型checkpoint路径')
    parser.add_argument('--start_idx', type=int, default=None, help='起始文件索引（可选）')
    parser.add_argument('--end_idx', type=int, default=None, help='结束文件索引（可选，包含）')
    parser.add_argument('--indices', type=str, default=None, help='指定文件索引列表，如 "46445,46446,46447,46448"')
    parser.add_argument('--model_type', type=str, default='zynq', choices=['orin', 'zynq'], help='模型类型')
    parser.add_argument('--is_super_resolution', type=_parse_bool_arg, default=True, help='启用超分辨率模式（默认True）')
    parser.add_argument('--threshold', type=float, default=0.5, help='输出阈值')
    parser.add_argument('--visualization', type=_parse_bool_arg, default=False, help='启用点云可视化（默认False）')
    parser.add_argument('--vis_save_dir', type=str, default="./output/radar_front_bottom_sr_visual", help='可视化图片保存目录')
    parser.add_argument('--save_output', type=_parse_bool_arg, default=True, help='保存输出PCD文件（默认True）')
    parser.add_argument('--occ_gt_dir', type=str, default='/data/dataset/PointSuperResolution/SR_Point_610_0302to0306/Val/Occ', help='Occ真值目录')
    parser.add_argument('--enable_evaluation', type=_parse_bool_arg, default=False, help='启用点云真值评估')
    parser.add_argument('--eval_output_file', type=str, default='evaluation_report.json', help='评估报告保存路径')
    parser.add_argument('--use_aai', type=_parse_bool_arg, default=True, help='启用AAI角精度提升叠加（将原始雷达AAI结果叠加到超分点云）')
    parser.add_argument('--use_original_overlay', type=_parse_bool_arg, default=False,
                        help='直接叠加原始点云（跳过AAI处理，将原始雷达点云缩放到SR分辨率后直接合并）。'
                             'use_aai优先级高于此参数；仅当use_aai=False时生效。')
    parser.add_argument('--use_dynamic_static', type=_parse_bool_arg, default=True, help='使用动静态分离的AAI处理（与aai_test.py一致，默认True）')
    parser.add_argument('--dynamic_continuity_threshold', type=int, default=2, help='动态点云连续性判断阈值，越小保留越多点 (default 1)')
    parser.add_argument('--dynamic_neighbor_threshold', type=int, default=2, help='动态点云邻居差值阈值，越大保留越多点 (default 3)')
    parser.add_argument('--add_offset', type=_parse_bool_arg, default=True, help='对模型输入的RCS和Doppler加70偏置（与训练一致，默认True；ORIN密集模型始终不加）')
    parser.add_argument('--max_match_per_orig', type=int, default=0,
                        help='每个原始点最大被SR点匹配次数，0表示不限制 (default 0)。'
                             'RCS采用bin距离线性衰减，Doppler直接赋值。')
    parser.add_argument('--enable_reuse_decay', type=_parse_bool_arg, default=False,
                        help='是否启用复用次数衰减（能量均分）。True时同一原始点被N个SR点复用，'
                             '每个SR点RCS衰减为1/N；False时仅bin距离线性衰减，不复用衰减 (default True)。')
    parser.add_argument('--add_is_sr', type=_parse_bool_arg, default=True,
                        help='在输出PCD中追加 is_sr 字段（1=模型SR生成，0=AAI叠加的原始点云）。'
                             '不指定该参数时输出与原格式一致（默认False）。')
    parser.add_argument('--use_neighborhood_filling', type=_parse_bool_arg, default=True,
                        help='启用邻域填充模式（新路径：RCS 3轴邻域count加权平均，'
                             'doppler/AbsV 中位数代表法选值，3轴全空丢弃SR点 (default False)。'
                             'False时走原 match_and_assign_values 路径。')
    parser.add_argument('--preserve_original_points', type=_parse_bool_arg, default=False,
                        help='输出中逐点保留原始x/y/z和五维特征，再加入模型SR点 (default False)。')
    parser.add_argument('--sr_min_rcs', type=float, default=None,
                        help='模型SR点的最小RCS；仅在preserve_original_points=True时生效。')
    parser.add_argument('--sr_min_abs_v', type=float, default=None,
                        help='模型SR点的最小|AbsV|；仅在preserve_original_points=True时生效。')
    parser.add_argument('--sr_static_min_rcs', type=float, default=None,
                        help='低速SR点达到此RCS时可绕过|AbsV|门控。')
    parser.add_argument('--sr_min_range', type=float, default=None,
                        help='模型SR点的最小物理距离（米）。')
    parser.add_argument('--sr_max_range', type=float, default=None,
                        help='模型SR点的最大物理距离（米，右开区间）。')
    parser.add_argument('--sr_empty_voxel_size', type=float, nargs=2, default=None,
                        metavar=('VOXEL_X', 'VOXEL_Y'),
                        help='仅填充原始点未占用的XY网格，每格保留最高RCS的SR点。')
    parser.add_argument('--expand_dynamic_raw', type=_parse_bool_arg, default=False,
                        help='将近距高RCS动态原始回波扩展到纵向相邻空网格。')
    parser.add_argument('--raw_expand_min_abs_v', type=float, default=1.5)
    parser.add_argument('--raw_expand_min_rcs', type=float, default=10.0)
    parser.add_argument('--raw_expand_max_range', type=float, default=50.0)
    parser.add_argument('--raw_expand_voxel_size', type=float, nargs=2,
                        default=(0.25, 0.20), metavar=('VOXEL_X', 'VOXEL_Y'))
    parser.add_argument('--raw_expand_rcs_scale', type=float, default=1.0,
                        help='合成原始支持点的 RCS 缩放')
    parser.add_argument('--raw_expand_absv_scale', type=float, default=1.0,
                        help='合成原始支持点的 AbsV 缩放')
    parser.add_argument('--expand_dense_raw', type=_parse_bool_arg, default=False,
                        help='将近距慢速、同一检测网格内多回波的原始点纵向扩展。')
    parser.add_argument('--dense_expand_min_points', type=int, default=8)
    parser.add_argument('--dense_expand_min_rcs', type=float, default=5.0)
    parser.add_argument('--dense_expand_max_abs_v', type=float, default=0.5)
    parser.add_argument('--dense_expand_max_range', type=float, default=50.0)
    parser.add_argument('--dense_expand_adaptive_axis', type=_parse_bool_arg, default=False,
                        help='以邻近合格密集网格的PCA主轴选择纵向或横向扩展。')
    parser.add_argument('--dense_expand_axis_radius', type=float, default=3.0)
    parser.add_argument('--dense_expand_min_axis_ratio', type=float, default=1.0)
    parser.add_argument('--dense_expand_lateral_steps', type=int, default=1,
                        help='PCA横向密集簇扩展的最大网格步数')
    parser.add_argument('--dense_expand_lateral_min_ratio', type=float, default=1.0,
                        help='启用额外横向步数所需的PCA各向异性比')
    parser.add_argument('--dense_expand_keep_longitudinal', type=_parse_bool_arg, default=False,
                        help='PCA选择横向扩展时，同时保留纵向邻格形成十字支持。')
    parser.add_argument('--dense_expand_require_adaptive_axis', type=_parse_bool_arg, default=False,
                        help='仅保留通过PCA横向门控的密集慢速种子。')
    parser.add_argument('--bridge_dense_raw', type=_parse_bool_arg, default=False,
                        help='在高各向异性横向慢速密集种子之间填充内部空网格。')
    parser.add_argument('--dense_bridge_max_gap', type=float, default=1.5)
    parser.add_argument('--dense_bridge_min_axis_ratio', type=float, default=10.0)
    args = parser.parse_args()

    # 同步动态点云AAI参数到全局变量（aai_frame3_dynamic_static内部读取）
    global DYNAMIC_CONTINUITY_THRESHOLD, DYNAMIC_NEIGHBOR_THRESHOLD
    DYNAMIC_CONTINUITY_THRESHOLD = args.dynamic_continuity_threshold
    DYNAMIC_NEIGHBOR_THRESHOLD = args.dynamic_neighbor_threshold

    IS_SUPER_RESOLUTION = args.is_super_resolution
    ENABLE_VISUALIZATION = args.visualization
    SAVE_OUTPUT = args.save_output
    ENABLE_EVALUATION = args.enable_evaluation and args.occ_gt_dir is not None

    # 超分辨率倍数（根据模式动态设置）
    if IS_SUPER_RESOLUTION:
        SR_RATE1 = 4  # Range和Azimuth: 4倍超分
        SR_RATE2 = 2  # Elevation: 2倍超分
    else:
        SR_RATE1 = 2  # u8模式: 2倍
        SR_RATE2 = 1  # u8模式: 1倍

    # 从参数获取配置
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    MODEL_PATH = Path(args.model_path)
    MODEL_TYPE = args.model_type
    THRESHOLD = args.threshold
    VIS_SAVE_DIR = args.vis_save_dir
    USE_AMP = False

    # 创建输出目录（仅当需要保存输出时）
    if SAVE_OUTPUT:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    if ENABLE_VISUALIZATION and VIS_SAVE_DIR:
        os.makedirs(VIS_SAVE_DIR, exist_ok=True)

    # 设置计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"可视化: {ENABLE_VISUALIZATION}, 保存目录: {VIS_SAVE_DIR}")
    print(f"保存输出PCD: {SAVE_OUTPUT}")
    print(f"启用评估: {ENABLE_EVALUATION}, Occ真值目录: {args.occ_gt_dir}")
    print(f"启用AAI角精度提升叠加: {args.use_aai}")
    print(f"直接叠加原始点云: {args.use_original_overlay} (仅当use_aai=False时生效)")
    print(f"AAI动静态分离模式: {args.use_dynamic_static}")
    print(f"动态点云AAI: continuity_threshold={DYNAMIC_CONTINUITY_THRESHOLD}, neighbor_threshold={DYNAMIC_NEIGHBOR_THRESHOLD}")
    print(f"输入偏置(+70): {args.add_offset}  (ORIN密集模型始终不加)")
    print(f"RCS衰减: bin距离线性衰减, Doppler直接赋值; max_match_per_orig={args.max_match_per_orig} (0=不限制)")
    print(f"复用次数衰减(能量均分): {args.enable_reuse_decay}")
    print(f"输出PCD追加 is_sr 字段: {args.add_is_sr} (1=模型SR生成, 0=AAI叠加)")

    # 评估结果累积
    eval_results = [] if ENABLE_EVALUATION else None

    # 加载模型（使用 run_inference 模块的 load_model）
    model = load_model(
        MODEL_PATH, device,
        model_type=MODEL_TYPE,
        is_super_resolution=IS_SUPER_RESOLUTION,
        use_amp=USE_AMP
    )

    # 获取所有 PCD 文件。递归模式保留相对目录结构，避免不同时间目录的
    # 同名文件在输出目录相互覆盖；如果根目录下没有直接 PCD，也自动尝试递归。
    input_root = Path(INPUT_DIR).expanduser().resolve()
    pcd_files = discover_pcd_files(str(input_root), recursive=args.recursive)
    total_files = len(pcd_files)
    print(f"发现 {total_files} 个PCD文件（输入根目录: {input_root}）")
    if total_files == 0:
        raise FileNotFoundError(f"目录中没有找到 PCD 文件: {input_root}")

    # 解析文件索引
    if args.indices:
        # 从逗号分隔字符串解析索引列表
        specified_indices = [int(i) for i in args.indices.split(',')]
        # 过滤有效索引
        valid_indices = [i for i in specified_indices if 0 <= i < total_files]
        if len(valid_indices) != len(specified_indices):
            invalid = [i for i in specified_indices if i not in valid_indices]
            print(f"警告：跳过无效索引: {invalid}")
        process_indices = valid_indices
    elif args.start_idx is not None or args.end_idx is not None:
        # 使用范围模式
        start_idx = args.start_idx if args.start_idx is not None else 0
        end_idx = args.end_idx if args.end_idx is not None else total_files - 1
        # 确保范围有效
        start_idx = max(0, min(start_idx, total_files - 1))
        end_idx = max(0, min(end_idx, total_files - 1))
        process_indices = list(range(start_idx, end_idx + 1))
    else:
        # 处理所有文件
        process_indices = list(range(total_files))

    # 只在指定了索引范围或列表时打印，处理所有文件时不打印（避免打印长列表）
    if args.indices is not None or args.start_idx is not None or args.end_idx is not None:
        print(f"待处理文件索引: {process_indices}")
    print(f"配置: model_type={MODEL_TYPE}, threshold={THRESHOLD}, sr_rate1={SR_RATE1}, sr_rate2={SR_RATE2}")

    # 处理指定文件
    for idx in process_indices:
        pcd_path = pcd_files[idx]
        pcd_file = pcd_path.name
        print(f"\n[{idx}/{total_files-1}] 处理文件: {pcd_file}")
        input_path = str(pcd_path)

        if SAVE_OUTPUT:
            # 生成输出文件名（添加_SR后缀）
            base_name = os.path.splitext(pcd_file)[0]
            output_filename = f"{base_name}_SR.pcd"
            try:
                relative_parent = pcd_path.parent.relative_to(input_root)
            except ValueError:
                relative_parent = Path('.')
            output_path = str(Path(OUTPUT_DIR) / relative_parent / output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = None  # 不保存输出文件

        # 查找对应的Occ真值文件
        occ_gt_path = None
        if ENABLE_EVALUATION and args.occ_gt_dir:
            # 尝试多种文件名匹配方式
            base_name = os.path.splitext(pcd_file)[0]
            # 方式1: 直接匹配（同名 + _coord.bin）
            occ_candidate1 = os.path.join(args.occ_gt_dir, f"{base_name}_coord.bin")
            # 方式2: 去除可能的后缀后匹配
            if base_name.endswith('_unknown_4d_radar_front'):
                base_name_stem = base_name.replace('_unknown_4d_radar_front', '')
                occ_candidate2 = os.path.join(args.occ_gt_dir, f"{base_name_stem}_coord.bin")
            else:
                occ_candidate2 = None

            if os.path.exists(occ_candidate1):
                occ_gt_path = occ_candidate1
            elif occ_candidate2 and os.path.exists(occ_candidate2):
                occ_gt_path = occ_candidate2
            else:
                print(f"  [警告] 未找到Occ真值文件: {base_name}")

        try:
            process_pcd_file(
                input_path, output_path, model, device,
                model_type=MODEL_TYPE,
                threshold=THRESHOLD,
                sr_rate1=SR_RATE1,
                sr_rate2=SR_RATE2,
                use_amp=USE_AMP,
                enable_visualization=ENABLE_VISUALIZATION,
                vis_save_dir=VIS_SAVE_DIR,
                occ_gt_path=occ_gt_path,
                eval_results=eval_results,
                use_aai=args.use_aai,
                use_original_overlay=args.use_original_overlay,
                use_dynamic_static=args.use_dynamic_static,
                add_offset=args.add_offset,
                max_match_per_orig=args.max_match_per_orig,
                enable_reuse_decay=args.enable_reuse_decay,
                add_is_sr=args.add_is_sr,
                use_neighborhood_filling=args.use_neighborhood_filling,
                preserve_original_points=args.preserve_original_points,
                sr_min_rcs=args.sr_min_rcs,
                sr_min_abs_v=args.sr_min_abs_v,
                sr_static_min_rcs=args.sr_static_min_rcs,
                sr_min_range=args.sr_min_range,
                sr_max_range=args.sr_max_range,
                sr_empty_voxel_size=args.sr_empty_voxel_size,
                expand_dynamic_raw=args.expand_dynamic_raw,
                raw_expand_min_abs_v=args.raw_expand_min_abs_v,
                raw_expand_min_rcs=args.raw_expand_min_rcs,
                raw_expand_max_range=args.raw_expand_max_range,
                raw_expand_voxel_size=args.raw_expand_voxel_size,
                raw_expand_rcs_scale=args.raw_expand_rcs_scale,
                raw_expand_absv_scale=args.raw_expand_absv_scale,
                expand_dense_raw=args.expand_dense_raw,
                dense_expand_min_points=args.dense_expand_min_points,
                dense_expand_min_rcs=args.dense_expand_min_rcs,
                dense_expand_max_abs_v=args.dense_expand_max_abs_v,
                dense_expand_max_range=args.dense_expand_max_range,
                dense_expand_adaptive_axis=args.dense_expand_adaptive_axis,
                dense_expand_axis_radius=args.dense_expand_axis_radius,
                dense_expand_min_axis_ratio=args.dense_expand_min_axis_ratio,
                dense_expand_lateral_steps=args.dense_expand_lateral_steps,
                dense_expand_lateral_min_ratio=args.dense_expand_lateral_min_ratio,
                dense_expand_keep_longitudinal=args.dense_expand_keep_longitudinal,
                dense_expand_require_adaptive_axis=args.dense_expand_require_adaptive_axis,
                bridge_dense_raw=args.bridge_dense_raw,
                dense_bridge_max_gap=args.dense_bridge_max_gap,
                dense_bridge_min_axis_ratio=args.dense_bridge_min_axis_ratio
            )
        except Exception as e:
            print(f"\n处理错误 {pcd_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 输出评估汇总报告
    if eval_results and len(eval_results) > 0:
        avg_acc = np.mean([r['acc'] for r in eval_results])
        avg_iou = np.mean([r['iou'] for r in eval_results])
        avg_ppv = np.mean([r['ppv'] for r in eval_results])
        avg_total_points = np.mean([r['total_points'] for r in eval_results])
        avg_hit_voxels = np.mean([r['hit_voxels'] for r in eval_results])

        print("\n" + "=" * 50)
        print("评估汇总报告")
        print("=" * 50)
        print(f"总评估帧数: {len(eval_results)}")
        print(f"平均ACC（点云准确率）: {avg_acc:.4f}")
        print(f"平均IoU（网格覆盖率）: {avg_iou:.4f}")
        print(f"平均PPV（每网格点密度）: {avg_ppv:.4f}")
        print(f"平均每帧有效点数: {avg_total_points:.1f}")
        print(f"平均每帧命中网格数: {avg_hit_voxels:.1f}")
        print("=" * 50)

        # 保存详细评估报告到JSON文件
        report = {
            'summary': {
                'total_frames': len(eval_results),
                'avg_acc': float(avg_acc),
                'avg_iou': float(avg_iou),
                'avg_ppv': float(avg_ppv),
                'avg_total_points': float(avg_total_points),
                'avg_hit_voxels': float(avg_hit_voxels)
            },
            'frames': eval_results
        }
        with open(args.eval_output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"评估报告已保存: {args.eval_output_file}")

    print("\n处理完成!")


if __name__ == "__main__":
    main()
