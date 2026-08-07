# 4D Radar PKL Web Visualizer

该工具直接读取 `infos_test_1000.pkl` 和本地二进制 PCD，不会修改原始数据。

展示内容：

- 前广图像：`CAMERA_FRONT_WIDE`，对应本地 `camera_front_wide`
- 4D Radar：`RADAR_FRONT`，界面显示名为 `radar_front`
- ATX 激光雷达：`LIDAR_FRONT_2`，界面显示名为 `lidar_front_2`
- EM4 激光雷达：`LIDAR_FRONT`，界面显示名为 `lidar_front`
- PKL 中的 3D 标注真值
- Radar 的 Doppler、RCS、Power、AbsV、Vx、Vy、Range 等维度着色
- 将 Radar、ATX、EM4 和真值框运动补偿到前广 FW 相机时间
- 支持按 `frame_id` 搜索可视化帧，并提供 80、150、200、300 米显示范围
- BEV 点云视图支持按钮和鼠标滚轮缩放，缩放范围为 `0.5x ~ 8x`
- 鼠标滚轮围绕光标缩放，按住左键可拖拽平移；复位按钮同时恢复倍率和视图中心

图像文件为 `3840x2160`，PKL 相机内参对应 `1920x1080` 标定分辨率。工具会自动缩放投影坐标。
前端展示图像限制为 `1280x720`，ATX 和 EM4 点云各抽样至最多 3 万点，原始数据不会被修改。

时间同步规则：

- FW 相机目标时间：从 `CAMERA_FRONT_WIDE` 文件名提取
- Radar：使用 PKL 中的 Radar 时间戳和 `imu2world`
- ATX：使用 PCD 逐点时间戳进行去畸变
- EM4：使用 PCD 帧时间戳
- 真值框：从 PKL 帧时间补偿至 FW 相机时间

最终输出坐标系为 `body_flu@camera_front_wide_timestamp`：

- 点云 `x/y/z` 与真值框均位于 FW 相机时刻的 body FLU
- Radar `Vx/Vy` 会旋转到该 body FLU；原始 Radar 轴速度保留为 `Vx_sensor/Vy_sensor`
- 图像投影时才临时使用 `body -> camera` 变换

## 启动

在仓库根目录执行：

```bash
python3 tools/radar_visualizer/server.py
```

浏览器打开：

```text
http://127.0.0.1:8765
```

指定其他数据目录或 PKL：

```bash
python3 tools/radar_visualizer/server.py \
  --data-root /path/to/data \
  --pkl /path/to/infos.pkl \
  --port 8765
```

叠加检测或跟踪结果：

```bash
python3 tools/radar_visualizer/server.py \
  --overlay-json output/demo_overlay.json
```

`overlay-json` 支持按 `frame_id`、`sequence_id` 或 PKL 索引组织。每个目标至少
需要 7 维 3D 框 `[x, y, z, length, width, height, yaw]`：

```json
{
  "frames": {
    "af12b3a1-7754-4251-b514-79eb76f59f56": [
      {
        "name": "Car",
        "score": 0.82,
        "source": "det",
        "box": [42.0, 1.2, 0.8, 4.5, 1.9, 1.7, 0.02]
      },
      {
        "name": "Truck",
        "track_id": "trk-17",
        "score": 0.76,
        "source": "track",
        "box": [80.0, -3.4, 1.4, 7.8, 2.6, 3.4, -0.1]
      }
    ]
  }
}
```

前端会将 `source=det` 画成粉色虚线框，将带 `track_id` 或 `source=track` 的
目标画成黄色实线框，并同时叠加到 BEV 与前广图像上。

依赖：Python 3、Flask、NumPy、Pillow。
