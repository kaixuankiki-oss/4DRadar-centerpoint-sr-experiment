# 4D Radar 3D Detection Project Overview

本项目面向车载 4D 毫米波雷达的神经网络 3D 目标检测，目标是在明确的
ODD（运行设计域）和统一评测协议下，使纯雷达方案逐步接近 Hesai ATX
LiDAR 方案的任务级检测性能。

当前阶段的重点不是直接设计新网络，而是建立：

1. 可复现的 LiDAR 与 4D Radar 检测基线；
2. 同车、同步、同标定的数据与评测闭环；
3. 点云路线和原始雷达张量路线的双轨研发体系；
4. 能够量化“达到同等水平”的验收指标。

详细计划见 [项目章程](project-charter.md)。

## 推荐的首个工程基线

- 框架：OpenPCDet
- LiDAR 对照：ATX 点云 + CenterPoint
- Radar 基线：多帧 4D Radar 点云 + PointPillars / CenterPoint
- Radar 点特征：`x, y, z, doppler, rcs, snr, timestamp, sensor_id`
- 训练监督：自采人工真值 + LiDAR teacher 蒸馏
- 推理输入：纯 4D Radar

## 当前阶段

`Phase 0`：明确产品 ODD、硬件边界、数据格式、评测协议和团队职责。
