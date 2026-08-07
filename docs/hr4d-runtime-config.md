# HR-4D 运行路径配置

项目运行路径统一定义在：

```text
tools/cfgs/path_configs/hr4d_paths.yaml
```

默认配置使用相对仓库根目录的路径：

```yaml
OUTPUT_ROOT: output

DATA_CONFIG:
    DATA_PATH: data/1000_original_data
    PKL_PATH: data/1000_original_data
    STORAGE:
        TYPE: file
        INFO_TYPE: file
```

因此本地仓库和远程服务器只要保持以下结构，就可以使用同一份配置：

```text
4DRadar/
├── data/
│   └── 1000_original_data/
├── output/
├── pcdet/
└── tools/
```

路径配置由 `tools/cfgs/radar_models/second_radar.yaml` 继承。模型配置和数据集
配置中不得再写个人电脑或服务器的绝对路径。

临时使用其他数据目录时，可以通过训练命令覆盖：

```bash
python train.py \
    --cfg_file cfgs/radar_models/second_radar.yaml \
    --set DATA_CONFIG.DATA_PATH /absolute/path/to/1000_original_data \
          DATA_CONFIG.PKL_PATH /absolute/path/to/pkl_root
```

`DATA_PATH` 表示原始数据根目录，`PKL_PATH` 表示 pkl 根目录。绝对路径会直接使用；相对路径始终相对仓库根目录解析。

## 存储系统选择

Radar 训练数据现在支持两种读取方式：

- `file`：默认模式，继续读取当前本地/挂载文件系统。
- `tos`：通过火山引擎 TOS Python SDK 读取对象存储中的 PCD 点云。
  info pkl 可以继续从本地读取，也可以通过完整 `tos://...pkl` 路径从
  TOS 读取。

TOS 模式需要安装可选依赖：

```bash
pip install tos
```

并在运行环境中配置访问凭证：

```bash
export TOS_AK=your_access_key
export TOS_SK=your_secret_key
export TOS_ENDPOINT=tos-cn-shanghai.ivolces.com
export TOS_REGION=cn-shanghai
```

训练时可以直接覆盖配置切换存储系统。TOS 数据根目录按当前火山对象
存储位置设置为：

```text
tos://perception-result/datasets/4d_data/original_data/
```

当前 pkl 存储位置为：

```text
tos://perception-result/pkls/4d_pkls/202606_12W_extend/
```

其中训练集 pkl 为 `infos_train.pkl`、`infos_train_bad_demotion.pkl`，
测试集 pkl 为 `infos_test.pkl`、`infos_rainy.pkl`。

pkl 中原始路径会在训练时自动映射。映射时会找到
`parse-process-data-2`，截取其后的月份及后续路径，然后拼到
`DATA_CONFIG.DATA_PATH` 后面，不替换 `parsed` 等目录名。例如：

```text
/mnt/nas_02/obs02/his3userrw/parse-process-data-2/202601/F772Y8/20260120103500/parsed/radar_front_bottom/a.pcd
```

会读取为：

```text
tos://perception-result/datasets/4d_data/original_data/202601/F772Y8/20260120103500/parsed/radar_front_bottom/a.pcd
```

只切换原始数据为 TOS、info pkl 仍使用本地同一份文件时：

```bash
python train.py \
    --cfg_file cfgs/radar_models/centerpoint_radar.yaml \
    --set DATA_CONFIG.STORAGE.TYPE tos \
          DATA_CONFIG.STORAGE.INFO_TYPE file \
          DATA_CONFIG.STORAGE.TOS.BUCKET perception-result \
          DATA_CONFIG.DATA_PATH datasets/4d_data/original_data \
          DATA_CONFIG.PKL_PATH /path/to/pkl_root \
          DATA_CONFIG.INFO_PATH.train "['infos_train.pkl']" \
          DATA_CONFIG.INFO_PATH.test "['infos_test.pkl']"
```

如果 info pkl 也上传到了 TOS，可以显式改成 TOS 读取。`DATA_PATH`
表示原始数据根目录，`PKL_PATH` 表示 pkl 根目录：

```bash
python train.py \
    --cfg_file cfgs/radar_models/centerpoint_radar.yaml \
    --set DATA_CONFIG.STORAGE.TYPE tos \
          DATA_CONFIG.STORAGE.INFO_TYPE tos \
          DATA_CONFIG.STORAGE.TOS.BUCKET perception-result \
          DATA_CONFIG.DATA_PATH datasets/4d_data/original_data \
          DATA_CONFIG.PKL_PATH pkls/4d_pkls/202606_12W_extend \
          DATA_CONFIG.INFO_PATH.train "['infos_train.pkl','infos_train_bad_demotion.pkl']" \
          DATA_CONFIG.INFO_PATH.test "['infos_test.pkl','infos_rainy.pkl']"
```

Radar SECOND 配置中的点云范围和体素大小还必须满足：

```text
(x_max - x_min) / voxel_x 可以被 16 整除
(y_max - y_min) / voxel_y 可以被 16 整除
```

否则检测头预测锚框数量会与目标分配器生成的锚框数量不一致。
