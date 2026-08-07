# HR4D Training Commands

本文档记录 HR-4D 当前训练代码下，两类常用数据存储方式的启动命令：

- 本地版：pkl 和原始数据都通过普通硬盘路径访问。
- TOS 版：pkl 和原始数据都在火山 TOS 上，通过训练配置切换访问方式。

以下命令默认在代码仓根目录已经准备好环境，并从 `tools` 目录启动训练。

## 1. 本地硬盘存储训练

适用环境：

- 本地容器
- 外网 L4 服务器
- 外网 A100 服务器
- 红区普通硬盘数据服务器

这种模式下，训练代码直接读取文件系统路径。`DATA_CONFIG.STORAGE.TYPE` 和 `DATA_CONFIG.STORAGE.INFO_TYPE` 都设为 `file`。

如果 pkl 文件不在默认数据目录下，训练时显式指定 pkl 路径：

```bash
cd /workspace/4DRadar/tools

python train.py \
  --cfg_file cfgs/radar_models/centerpoint_radar.yaml \
  --extra_tag file_base \
  --workers 8 \
  --set DATA_CONFIG.STORAGE.TYPE file \
        DATA_CONFIG.STORAGE.INFO_TYPE file \
        DATA_CONFIG.DATA_PATH /mnt/nas_02 \
        DATA_CONFIG.PKL_PATH /path/to/pkl_root \
        DATA_CONFIG.INFO_PATH.train "['infos_train.pkl']" \
        DATA_CONFIG.INFO_PATH.test "['infos_test.pkl']"
```

本地模式参数含义：

- `DATA_CONFIG.STORAGE.TYPE=file`：原始数据走普通文件系统。
- `DATA_CONFIG.STORAGE.INFO_TYPE=file`：pkl 文件走普通文件系统。
- `DATA_CONFIG.DATA_PATH`：本地原始数据根目录。例如 pkl 里记录 `obs02/...` 时，设置为 `/mnt/nas_02` 后会拼成 `/mnt/nas_02/obs02/...`。
- `DATA_CONFIG.PKL_PATH`：本地 pkl 根目录。
- `DATA_CONFIG.INFO_PATH.train/test`：训练和验证 pkl 文件，可使用绝对路径或相对 `PKL_PATH` 的路径。

## 2. pkl 和原始数据都在 TOS 的训练

适用环境：

- 红区火山服务器
- 能访问火山 TOS 的训练容器或 OpenPAI 任务

这种模式下，pkl 文件和 pkl 中记录的原始数据路径都通过 TOS 读取。训练代码会在运行时做路径适配，使同一份 pkl/yaml 不需要为文件存储和 TOS 存储分别改内容。

先准备 TOS SDK 和访问密钥：

```bash
pip install tos

export TOS_AK=<your_tos_access_key>
export TOS_SK=<your_tos_secret_key>
export TOS_ENDPOINT=tos-cn-shanghai.ivolces.com
export TOS_REGION=cn-shanghai
```

启动训练：

```bash
cd /workspace/4DRadar/tools

python train.py \
  --cfg_file cfgs/radar_models/centerpoint_radar.yaml \
  --extra_tag tos_full \
  --workers 8 \
  --set DATA_CONFIG.STORAGE.TYPE tos \
        DATA_CONFIG.STORAGE.INFO_TYPE tos \
        DATA_CONFIG.STORAGE.TOS.BUCKET perception-result \
        DATA_CONFIG.DATA_PATH datasets/4d_data/original_data \
        DATA_CONFIG.PKL_PATH pkls/4d_pkls/202606_12W_extend \
        DATA_CONFIG.INFO_PATH.train "['infos_train.pkl','infos_train_bad_demotion.pkl']" \
        DATA_CONFIG.INFO_PATH.test "['infos_test.pkl','infos_rainy.pkl']"
```

如果希望把 pkl 根目录和原始数据根目录都写成完整 TOS URI，也可以这样写：

```bash
cd /workspace/4DRadar/tools

python train.py \
  --cfg_file cfgs/radar_models/centerpoint_radar.yaml \
  --extra_tag tos_full \
  --workers 8 \
  --set DATA_CONFIG.STORAGE.TYPE tos \
        DATA_CONFIG.STORAGE.INFO_TYPE tos \
        DATA_CONFIG.DATA_PATH tos://perception-result/datasets/4d_data/original_data \
        DATA_CONFIG.PKL_PATH tos://perception-result/pkls/4d_pkls/202606_12W_extend \
        DATA_CONFIG.INFO_PATH.train "['infos_train.pkl','infos_train_bad_demotion.pkl']" \
        DATA_CONFIG.INFO_PATH.test "['infos_test.pkl','infos_rainy.pkl']"
```

OpenPAI 中建议只把代码包名和解压目录用变量统一管理。`${CODE_ARCHIVE%.tar.gz}` 会把 `4DRadar.tar.gz` 转成 `4DRadar`。
训练时需要修改的内容为：
1、CODE_ARCHIVE=4DRadar.tar.gz的压缩包名称；
2、EXP_NAME=centerpoint_minpts3实验记录保存的文件夹名称；
3、trian.py后面的训练指令，尤其是yaml文件。

```yaml
commands:
  - set -euo pipefail
  - export CODE_ARCHIVE=4DRadar-HR-4D.tar.gz
  - export CODE_DIR="${CODE_ARCHIVE%.tar.gz}"
  - export EXP_NAME=centerpoint_base_epoch1
  - bash /mnt/data-vepfs/token/huoshan-tos.sh
  - mkdir -p /mnt/nas && cd /mnt/nas
  - rm -rf "${CODE_DIR}"
  - tosutil cp -r tos://e2e-training/code/${PAI_USER_NAME}/openpcdet/${CODE_ARCHIVE} .
  - tar -zxf "${CODE_ARCHIVE}"
  - source ~/miniconda/etc/profile.d/conda.sh
  - conda activate 4d
  - export TORCH_CUDA_ARCH_LIST="8.0"
  - export CUDA_HOME=/usr/local/cuda
  - ulimit -l unlimited
  - export OMP_NUM_THREADS=8
  - export MKL_NUM_THREADS=8
  - export TORCH_NUM_THREADS=8
  - export NCCL_IB_DISABLE=1
  - export NCCL_SOCKET_IFNAME=eth0
  - export GLOO_SOCKET_IFNAME=eth0
  - export TOS_AK=<your_tos_access_key>
  - export TOS_SK=<your_tos_secret_key>
  - export TOS_ENDPOINT=tos-cn-shanghai.ivolces.com
  - export TOS_REGION=cn-shanghai
  - cd "/mnt/nas/${CODE_DIR}"
  - pip install setuptools==58.2.0
  - python setup.py develop
  - cd tools
  - mkdir -p "/mnt/nas/${CODE_DIR}/output/${EXP_NAME}"
  - |
    trap 'echo "准备回传 TOS..."; tosutil cp -r -u /mnt/nas/${CODE_DIR}/output/${EXP_NAME} tos://e2e-training/product/${PAI_USER_NAME}/${EXP_NAME}/' EXIT
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
      --nproc_per_node=8 \
      train.py \
      --launcher pytorch \
      --cfg_file cfgs/radar_models/centerpoint_radar.yaml \
      --epochs 1 \
      --batch_size 128 \
      --logger_iter_interval 500 \
      --extra_tag "/mnt/nas/${CODE_DIR}/output/${EXP_NAME}" \
      --set DATA_CONFIG.STORAGE.TYPE tos \
            DATA_CONFIG.STORAGE.INFO_TYPE tos \
            DATA_CONFIG.STORAGE.TOS.BUCKET perception-result \
            DATA_CONFIG.DATA_PATH datasets/4d_data/original_data \
            DATA_CONFIG.PKL_PATH pkls/4d_pkls/202606_12W_extend \
            DATA_CONFIG.INFO_PATH.train "['infos_train.pkl','infos_train_bad_demotion.pkl']" \
            DATA_CONFIG.INFO_PATH.test "['infos_test.pkl','infos_rainy.pkl']"
```

TOS 模式参数含义：

- `DATA_CONFIG.STORAGE.TYPE=tos`：原始数据走 TOS。
- `DATA_CONFIG.STORAGE.INFO_TYPE=tos`：pkl 文件也走 TOS。
- `DATA_CONFIG.STORAGE.TOS.BUCKET=perception-result`：TOS bucket 名称。
- `DATA_CONFIG.DATA_PATH`：TOS 上原始数据根目录，可以是 bucket 内相对路径，也可以是完整 `tos://bucket/prefix`。
- `DATA_CONFIG.PKL_PATH`：TOS 上 pkl 根目录，可以是 bucket 内相对路径，也可以是完整 `tos://bucket/prefix`。
- `DATA_CONFIG.INFO_PATH.train/test`：pkl 文件位置。使用相对路径时，相对 `DATA_CONFIG.PKL_PATH`。

当前 TOS 数据位置：

- 原始数据：`tos://perception-result/datasets/4d_data/original_data`
- pkl 根目录：`tos://perception-result/pkls/4d_pkls/202606_12W_extend`
- 训练集 pkl：`infos_train.pkl`、`infos_train_bad_demotion.pkl`
- 测试集 pkl：`infos_test.pkl`、`infos_rainy.pkl`

当前代码会把 pkl 中原始文件存储路径自动映射到 TOS 路径。例如 pkl 中类似：

```text
/mnt/nas_02/obs02/his3usercw/parse-process-data-2/202601/F772Y8/20260131170000/parsed/camera_front_narrow/...
```

会在 TOS 模式下映射为：

```text
tos://perception-result/datasets/4d_data/original_data/202601/F772Y8/20260131170000/parsed/camera_front_narrow/...
```

映射规则只截取 pkl 路径中 `parse-process-data-2` 后面的月份及后续路径，再拼到 `DATA_CONFIG.DATA_PATH` 后面。

## 3. test.py 推理和评测

`test.py` 和 `train.py` 使用同一套 dataloader 与 `DATA_CONFIG.STORAGE` 配置，因此也支持 TOS。区别是模型 checkpoint 仍通过 `--ckpt` 从当前文件系统路径读取；如果 checkpoint 在 TOS 上，先用平台命令或 `tosutil` 下载/挂载到本地路径。

本地硬盘存储评测：

```bash
cd /workspace/4DRadar/tools

CUDA_VISIBLE_DEVICES=0 python test.py \
  --cfg_file cfgs/radar_models/centerpoint_radar.yaml \
  --ckpt /path/to/checkpoint_epoch_x.pth \
  --batch_size 16 \
  --workers 4 \
  --extra_tag file_eval \
  --eval_tag epoch_x \
  --set DATA_CONFIG.STORAGE.TYPE file \
        DATA_CONFIG.STORAGE.INFO_TYPE file \
        DATA_CONFIG.DATA_PATH /mnt/nas_02 \
        DATA_CONFIG.PKL_PATH /path/to/pkl_root \
        DATA_CONFIG.INFO_PATH.test "['infos_test.pkl','infos_rainy.pkl']"
```

TOS 存储评测：

```bash
cd /workspace/4DRadar/tools

export TOS_AK=<your_tos_access_key>
export TOS_SK=<your_tos_secret_key>
export TOS_ENDPOINT=tos-cn-shanghai.ivolces.com
export TOS_REGION=cn-shanghai

CUDA_VISIBLE_DEVICES=0 python test.py \
  --cfg_file cfgs/radar_models/centerpoint_radar.yaml \
  --ckpt /path/to/checkpoint_epoch_x.pth \
  --batch_size 16 \
  --workers 4 \
  --extra_tag tos_eval \
  --eval_tag epoch_x_tos \
  --set DATA_CONFIG.STORAGE.TYPE tos \
        DATA_CONFIG.STORAGE.INFO_TYPE tos \
        DATA_CONFIG.STORAGE.TOS.BUCKET perception-result \
        DATA_CONFIG.DATA_PATH datasets/4d_data/original_data \
        DATA_CONFIG.PKL_PATH pkls/4d_pkls/202606_12W_extend \
        DATA_CONFIG.INFO_PATH.test "['infos_test.pkl','infos_rainy.pkl']"
```

多卡评测时同样使用 `torchrun`，并显式传入 `--launcher pytorch`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 test.py \
  --launcher pytorch \
  --cfg_file cfgs/radar_models/centerpoint_radar.yaml \
  --ckpt /path/to/checkpoint_epoch_x.pth \
  --batch_size 64 \
  --workers 8 \
  --extra_tag tos_eval_4gpu \
  --eval_tag epoch_x_tos \
  --set DATA_CONFIG.STORAGE.TYPE tos \
        DATA_CONFIG.STORAGE.INFO_TYPE tos \
        DATA_CONFIG.STORAGE.TOS.BUCKET perception-result \
        DATA_CONFIG.DATA_PATH datasets/4d_data/original_data \
        DATA_CONFIG.PKL_PATH pkls/4d_pkls/202606_12W_extend \
        DATA_CONFIG.INFO_PATH.test "['infos_test.pkl','infos_rainy.pkl']"
```

`test.py` 的 `--batch_size` 和训练一致，命令行传入的是总 batch size。多卡时程序会除以 GPU 数作为单卡 batch size，例如 4 卡 `--batch_size 64` 表示单卡 batch size 为 16。

## 4. 存储方式切换速查
切换原则：

- 只改训练启动参数即可切换存储系统。
- 本地训练使用 `TYPE=file` 和 `INFO_TYPE=file`。
- pkl 也在 TOS 时，必须同时设置 `TYPE=tos` 和 `INFO_TYPE=tos`。
- `DATA_PATH` 始终表示原始数据根目录。
- `PKL_PATH` 始终表示 pkl 根目录。
- pkl 不在 `PKL_PATH` 下时，`INFO_PATH` 使用绝对文件路径或完整 `tos://...pkl`。

实际训练前需要把本地模式示例中的 `/path/to/...` 替换为当前实验使用的真实本地 pkl 路径；TOS 模式已按当前火山数据位置填写。
