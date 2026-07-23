# AcupointMMNet_Hand 项目说明

本项目用于针灸机器人场景下的多任务检测。
模型同时学习两类目标：

- 关键点回归（穴位坐标）
- 分割任务（反射区/区域语义）

当前仓库包含训练脚本、测试脚本、模型定义、数据读取代码，以及示例数据目录。

## 1. 项目结构

下面是最常用目录的作用说明：

- `train.py`：训练入口。
- `test.py`：可视化测试入口。
- `opts.py`：命令行参数定义。
- `src/engine/`：训练/评估/可视化共用的运行时与流程编排。
- `src/model/`：模型结构和损失函数。
- `src/dataset/`：数据集工厂和数据读取。
- `src/tools/`：结果汇总与离线辅助脚本。
- `acuSim/`：AcuSim 数据与数据处理脚本。
- `acuSim_tiny_smoke/`：小规模烟雾测试数据，可快速验证流程。

## 2. 数据集与格式

本项目支持以下数据集标识：

- `--dataset hand`
- `--dataset cervicocranial`
- `--dataset acusim`

注意：`opts.py` 中的默认值是 `luojiassr`，与当前代码不匹配。实际运行时必须手动传入上面的可用值。

### 2.1 AcuSim 目录要求

默认数据根目录为：

`acuSim/dataset/main/dataset`

目录结构示例：

```text
acuSim/dataset/main/dataset/
├── map.txt
├── train/
│   ├── image/img_512/
│   └── label/label/
└── val/
    ├── image/img_512/
    └── label/label/
```

### 2.2 标注关键点说明

- 模型当前按 16 个关键点输出。
- 如果不传 `--cervico_keypoints`，程序默认读取 `map.txt` 前 16 项。
- JSON 中关键点字段读取规则：
	- 名称：`label[].name`
	- 坐标：`label[].coordinate.x`、`y`、`h`
	- 坐标范围会被裁剪到 `[0, 1]`

### 2.3 数据集获取说明

当前仓库没有把完整原始数据集一并上传到 GitHub。仓库中提供的是示例数据与小规模测试数据（例如 `acuSim_tiny_smoke`），以及数据读取/标注格式说明。

如果你需要完整的 AcuSim / hand / cervicocranial 数据，请在这个链接下载：
- acuSim.zip 数据下载链接：https://datadryad.org/dataset/doi:10.5061/dryad.zs7h44jkz 
- AcuSim 数据目录：`acuSim/dataset/main/dataset`
- 小规模快速验证数据：`acuSim_tiny_smoke`


## 3. 环境安装

### 3.1 克隆仓库

```bash
git clone https://gitcode.com/qq_38063965/AcupointMMNet_Hand.git
cd AcupointMMNet_Hand
```

### 3.2 创建环境（推荐）

```bash
conda create -n acupointnet python=3.10 -y
conda activate acupointnet
```

### 3.3 安装依赖

先安装基础依赖：

```bash
pip install -r requirements.txt
```

再安装训练和可视化常用扩展依赖：

```bash
pip install numpy scipy opencv-python matplotlib scikit-learn scikit-image
pip install mediapipe thop torchmetrics einops
```

## 4. 训练

`train.py` 使用分布式初始化。推荐用 `torchrun` 启动，即使只用 1 张卡也这样做。

### 4.1 单卡训练示例

```bash
torchrun --nproc_per_node=1 train.py \
	--dataset acusim \
	--exp_id exp_acusim_001 \
	--gpus 0 \
	--num_epochs 50 \
	--batch_size 4 \
	--cervico_dataset_root acuSim/dataset/main/dataset \
	--cervico_image_subdir img_512
```

### 4.2 快速流程自检（小数据）

```bash
torchrun --nproc_per_node=1 train.py \
	--dataset acusim \
	--exp_id smoke_run \
	--gpus 0 \
	--num_epochs 2 \
	--batch_size 2 \
	--cervico_dataset_root acuSim_tiny_smoke \
	--cervico_image_subdir img_512
```

### 4.3 结果保存位置

训练输出保存在：

`../results/acupointmm/<exp_id>/`

也可以通过 `--output_root` 指定新的实验输出根目录。

目录内会包含：

- `config.json`：本次运行参数
- `ckpt_epoch_*.pth`：模型权重
- 日志文件

## 5. 测试与可视化

```bash
python test.py --dataset acusim --exp_id test_vis --gpus 0 --load_model <模型权重路径>
```

说明：

- `test.py` 中存在固定 GPU 与分布式初始化写法。
- 若你的机器卡号或端口不同，请先按本地环境调整再运行。

### 5.1 生成技术交付报告

评估结束后，可通过下面的命令把结果汇总成一份可直接用于汇报的 Markdown 文件：

```bash
python src/tools/summarize_results.py --results_dir ../results/acupointmm/<exp_id>
```

生成结果会写入：

- `../results/acupointmm/<exp_id>/summary_report.md`

## 6. 常见问题

### 6.1 报错 `Unsupported dataset`

原因：`--dataset` 传了无效值。

解决：改为 `hand`、`cervicocranial` 或 `acusim`。

### 6.2 报错找不到图像或标注目录

原因：`--cervico_dataset_root` 或 `--cervico_image_subdir` 不正确。

解决：检查目录是否存在，确认图像与 JSON 文件名一一对应。

### 6.3 报错 `exp_id null !!!`

原因：未传 `--exp_id`，默认值为 `default`。

解决：为每次实验设置唯一 `--exp_id`。

## 7. 引用

如果你在论文或项目中使用本仓库，请引用对应论文：

Zheng, Y., Liao, C., Zhang, H., & He, Q. (2025). Simultaneous Multimodal Detection of Hand Acupoints and Reflex Zones for Acupuncture Robots.

## 8. 交付模板

- [docs/technical_deliverable_cn.md](docs/technical_deliverable_cn.md)