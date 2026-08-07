# HR-4D 评测可视化工具使用说明与 Release Note

本文面向 HR-4D 团队同步伟康侧评测可视化工具的当前能力、使用方式和
release note。该工具用于 smoke / exploratory 阶段的定性诊断，帮助大家快速
定位模型预测与 GT 不一致的位置；它不替代正式评测指标，也不作为正式实验结论。

## 一句话说明

输入 OpenPCDet 推理结果 `result.pkl`、HR-4D infos 和原始数据目录后，工具会
生成一个静态 Web review bundle。页面里可以同时查看相机图像、Radar 点云、
ATX LiDAR、EM4 LiDAR、GT、预测框、图像投影和按 case 排序的 FP/FN/LOC
问题。

## 适用场景

- 快速查看某次推理结果是否跑通。
- 挑出高优先级 FN、FP 和定位异常 case。
- 检查问题帧中 Radar / ATX / EM4 点云对目标的支持情况。
- 对比预测框与 GT 在 3D、BEV 和图像投影上的差异。
- 为周报、阶段汇报和模型实验复盘提供可截图、可复现的定性证据。

## 当前示例数据

当前仓库内已验证的示例输入：

```text
infos:       data/1000_original_data/splits/hr4d_1000_v1/infos_test_200.pkl
prediction:  output/weikang_tracking/user_result.pkl
data root:   data/1000_original_data
```

当前示例导出摘要：

| Metric | Value |
| --- | ---: |
| Frames | 200 |
| GT boxes in eval region | 1409 |
| Predictions in eval region | 1437 |
| TP | 756 |
| FP | 681 |
| FN | 653 |
| LOC warnings | 31 |
| Selected review cases | 61 |
| Selected review frames | 40 |

## 生成可视化 Bundle

在服务器上使用已有容器 runner，避免重复创建容器或镜像：

```bash
/usr/local/bin/hr4d-run bash -lc '
cd /workspace/4DRadar
python tools/radar_visualizer/export_eval_visualization.py \
  --infos data/1000_original_data/splits/hr4d_1000_v1/infos_test_200.pkl \
  --data-root data/1000_original_data \
  --predictions output/weikang_tracking/user_result.pkl \
  --score-threshold 0.15 \
  --match-lateral-threshold 2.0 \
  --loc-warning-threshold 1.0 \
  --max-cases 80 \
  --max-frames 40 \
  --output-dir output/weikang_eval_review/user_result \
  --indent-json
'
```

输出目录：

```text
output/weikang_eval_review/user_result/index.html
output/weikang_eval_review/user_result/assets/index.json
output/weikang_eval_review/user_result/assets/frame_*.json
output/weikang_eval_review/user_result/assets/frame_*.jpg
output/weikang_eval_review/user_result/eval_diff.json
output/weikang_eval_review/user_result/eval_overlay.json
```

## 打开页面

在导出的 bundle 目录启动静态服务：

```bash
cd output/weikang_eval_review/user_result
python -m http.server 8899
```

浏览器打开：

```text
http://127.0.0.1:8899/index.html
```

如果服务跑在远端服务器上，需要先做端口转发，再从本地浏览器访问。

## 页面功能说明

### Case 列表

- `ALL`: 显示被选出的所有重点 review case。
- `FN`: 漏检 GT，优先用于检查模型没看到的目标。
- `FP`: 误检预测框，优先用于检查重复框、背景误检和类别混淆。
- `LOC`: 已找到目标但位置、朝向或尺寸误差较大的 case。

点击 case 后，页面会切换到对应帧，并在 3D / BEV / 图像中高亮相关 GT 或
prediction。

### 3D 主视图

3D 视图用于替代只看 BEV 的扁平检查方式，适合看高度、点云支撑、框体姿态和
前后遮挡关系。

常用控件：

- `ISO`: 斜视角，适合快速看整体 3D 关系。
- `TOP`: 俯视角，接近传统 BEV。
- `FRONT`: 前视角，适合看高度和横向偏差。
- `LEFT`: 侧视角，适合看纵向距离和高度。
- `FIT`: 根据当前帧点云范围自动收拢视角。
- `FOCUS`: 选中 case 后聚焦到相关目标附近。
- `ORBIT`: 鼠标拖拽旋转视角。
- `PAN`: 鼠标拖拽平移视角。
- `Yaw / Pitch / Zoom / Z`: 精细调整视角、距离和高度拉伸。

### BEV 辅助视图

BEV 用于快速检查平面位置关系。支持鼠标滚轮缩放、拖拽平移、range preset 和
图层开关。

### 图像投影

图像视图会叠加：

- Radar 点云投影。
- GT 3D box 投影。
- Prediction 3D box 投影。

如果图像投影与 3D / BEV 明显不一致，优先检查同步、标定、坐标系和 box yaw。

### Radar 点云颜色模式

Radar 点云支持以下颜色模式，并且同时作用于 3D、BEV 和图像投影：

| Mode | 用途 |
| --- | --- |
| `Radar color` | 固定黄色，适合只看点云覆盖范围 |
| `Radar RCS` | 按反射强度着色，低值蓝色，高值黄色 |
| `Radar Doppler` | 按多普勒速度着色，负值蓝色，零附近白色，正值红色 |
| `Radar AbsV` | 按速度幅值着色，低值青色，高值粉红色 |

选中 `RCS`、`Doppler` 或 `AbsV` 时，信息栏会显示该帧该字段的
`min / median / max`，用于解释当前颜色范围。

### 图层开关

可以单独开关：

- Radar
- ATX LiDAR
- EM4 LiDAR
- GT
- Prediction
- Image projection

建议先只开 Radar + GT + Prediction 看模型与 Radar 的关系，再逐步打开 ATX 和
EM4 判断是数据支撑问题还是模型后处理问题。

## 颜色语义

| 元素 | 颜色 / 含义 |
| --- | --- |
| Matched GT | 绿色 |
| FN GT | 红橙色 |
| Ignored GT | 灰色，不在评测区域内，仅作为上下文 |
| TP Prediction | 青色 |
| FP Prediction | 粉色 |
| Selected / LOC context | 黄色 |
| ATX LiDAR points | 蓝色 |
| EM4 LiDAR points | 紫色 |

## 建议 Review 流程

1. 先看 `FN`，记录远距离、稀疏点云、遮挡和类别相关的漏检模式。
2. 再看高分 `FP`，区分重复框、背景误检、类别混淆和 ROI 边界问题。
3. 对 `LOC` case 检查中心点、yaw、尺寸和高度误差。
4. 在 3D 视图里用 `ISO / FRONT / LEFT / FOCUS` 检查 BEV 看不清的问题。
5. 切换 Radar `RCS / Doppler / AbsV`，观察错误附近是否存在有效反射或速度线索。
6. 打开 ATX / EM4，判断 LiDAR GT 支撑与 Radar-only 输入之间的差异。
7. 在图像投影中检查同步、标定和朝向是否存在明显异常。
8. 输出结论时记录 `frame_id`、`eval_id`、case 类型和截图。

## Release Note

### 新增能力

- 新增 `eval_diff.py`，基于 HR-4D 椭圆中心距离协议计算 TP / FP / FN / LOC。
- 新增 `export_eval_visualization.py`，将 OpenPCDet `result.pkl` 导出为静态 Web
  review bundle。
- 新增按 case 排序的问题帧选择逻辑，优先导出高价值 FN、FP 和 LOC。
- 新增 3D 主视图，支持 PCL-style 视角切换、旋转、平移、缩放和选中目标聚焦。
- 新增 Radar、ATX、EM4 多源点云图层，支持与 GT / prediction 同屏对比。
- 新增 GT 与 prediction 的图像投影，用于检查同步、标定和朝向问题。
- 新增 Radar 点云 `RCS / Doppler / AbsV` 着色模式。
- 新增 ignored GT 灰色展示，避免把评测区域外 GT 误读为 FN。
- 新增离线 tracking 工具骨架和文档，后续可接入 DET+TRACK 对比视频。

### 主要文件

```text
tools/radar_visualizer/eval_diff.py
tools/radar_visualizer/export_eval_visualization.py
tools/radar_visualizer/EVAL_VISUALIZATION.md
tools/radar_visualizer/server.py
tools/radar_visualizer/export_fusion_video.py
tools/tracking/hr4d_offline_tracker.py
tools/tracking/README.md
tests/test_eval_visualization.py
docs/hr4d-eval-visualization-review.md
docs/hr4d-eval-visualization-release-note.md
```

### 验证结果

合入 `HR-4D` 前已运行：

```bash
python -m unittest -q \
  tests.test_package_version \
  tests.test_hr4d_eval \
  tests.test_radar_ego_doppler \
  tests.test_radar_pillar_vfe \
  tests.test_eval_visualization
```

期望结果：

```text
Ran 21 tests
OK
```

当前示例 bundle 已确认：

- `index.html` 可由静态 HTTP 服务打开。
- case 列表、过滤和选中行为可用。
- 3D / BEV / 图像三类视图可渲染。
- Radar / ATX / EM4 / GT / Prediction 图层可开关。
- Radar `RCS / Doppler / AbsV` 着色模式可切换。
- ignored GT 使用灰色展示。

## 已知限制

- 当前页面用于检测结果 review，尚未把 tracking IDSW、fragmentation 和轨迹连续性
  指标纳入统一面板。
- 静态 bundle 会包含选中帧的点云 JSON，帧数过多时文件体积会明显变大。
- 当前只导出高优先级帧用于页面交互，全量 case 仍保留在 `eval_diff.json`。
- 颜色范围是逐帧 min / median / max，不同帧之间颜色不一定可以直接做绝对比较。
- 该工具支持定性诊断和问题定位，不替代正式评测表格。

## 同步建议

- 龙博可以用该页面检查数据问题帧、同步和标定异常。
- 孙博可以用 FN / FP / LOC case 反推模型结构和预处理问题。
- 建永｜实验结果分析可以把截图与指标表结合，做错误类型归因。
- 伟康继续在该工具上扩展 DET+TRACK 对比、轨迹稳定性和视频导出。
- 泽林可以直接引用本 release note 的功能清单和限制说明做周报同步。
