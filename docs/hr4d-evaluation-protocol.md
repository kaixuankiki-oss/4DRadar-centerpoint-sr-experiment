# HR-4D 3D 检测评测协议

本协议以 nuScenes 的中心点距离匹配与置信度排序 AP 为基础，用于在同一批
数据和同一套真值上公平比较 4D Radar 与 ATX LiDAR 检测结果。

## 1. 坐标系与评测区域

- 使用车体 FLU 坐标系：`x` 向前，`y` 向左，`z` 向上。
- 中心点匹配只使用 BEV 平面的 `x, y`，不使用高度差。
- 纵向范围：`0 <= x <= 200m`。
- 横向范围：`-20m <= y <= 20m`。
- BEV 径向距离：不超过 `200m`，保证整体结果等于四个距离段的并集。
- 水平视场：相对车辆前向左右各 `40°`。
- GT 与预测框必须通过相同区域过滤。

最终评测区域是纵向、横向矩形与半径 `200m`、前向 `±40°` 扇形的交集。

## 2. 椭圆中心点距离

每个 GT 的椭圆长轴沿“自车原点到 GT 中心”的径向方向，短轴沿其横向方向。
设预测框与 GT 中心误差在径向和横向上的分量分别为 `e_r`、`e_t`，横向
匹配阈值为 `T`，则：

```text
sqrt((e_r / 2)^2 + e_t^2) < T
```

因此径向允许误差为 `2T`，横向允许误差为 `T`。默认横向阈值沿用
nuScenes 多阈值 AP 思路：

```text
T = 0.5m, 1.0m, 2.0m, 4.0m
```

评测按预测置信度从高到低执行，同类别、同帧内进行一对一贪心匹配。一个
GT 最多只能被一个预测框匹配。

## 3. 距离分段

目标距离使用 BEV 径向距离 `sqrt(x^2 + y^2)`，分别输出：

- 整体 `overall`
- `0-50m`
- `50-100m`
- `100-150m`
- `150-200m`

GT 与预测框均按照自身中心所在距离段过滤，避免跨距离段统计。

## 4. 输出指标

- 每类别、每距离段、每个椭圆阈值的 AP 与最大 Recall；
- 每类别、每距离段在所有阈值上的 mean AP；
- 每个距离段跨有效类别的 mAP；
- 整体 mAP；
- 在横向阈值 `2m` 下统计中心、径向、横向、尺寸和朝向误差；
- 当 GT 与预测均具有有效速度时，统计速度误差。

AP 计算沿用 nuScenes 的 101 点插值方式，默认忽略低于 `10%` 的 Recall 和
低于 `10%` 的 Precision。

当前不输出 NDS。nuScenes NDS 包含速度和属性等指标，而现有 HR-4D 标签的
速度均为无效值，且没有 nuScenes 属性标签；此时强行生成 NDS 会产生误导。

## 5. 代码接口

核心实现：

```text
pcdet/datasets/hr4d/hr4d_eval/evaluation.py
```

OpenPCDet 风格接口：

```python
from pcdet.datasets.hr4d.hr4d_eval import get_evaluation_results

result_str, result_dict = get_evaluation_results(
    gt_annos,
    pred_annos,
    class_names=['Car', 'Truck', 'Pedestrian', 'Cyclist'],
)
```

支持当前 HR-4D GT 字段 `names`、`boxes_3d`，以及 OpenPCDet 预测字段
`name`、`boxes_lidar`、`score`。

也可以直接评测 OpenPCDet 输出的 `result.pkl`：

```bash
python tools/eval_utils/eval_hr4d.py \
    --gt_infos data/1000_original_data/infos_test_1000.pkl \
    --predictions output/.../result.pkl \
    --classes Car Truck Pedestrian Cyclist \
    --output_json output/.../hr4d_metrics.json
```
