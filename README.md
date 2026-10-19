# R-TPT-Adapter：钢铁表面缺陷小样本分类

[English](README.en.md) · [历史实验记录](docs/历史实验记录.md) · [技术实现](docs/伽马对比度缓存引导R-TPT技术实现.md)

## 项目简介

本项目在 NEU-CLS 六类钢铁表面缺陷上结合 CLIP RN50、16-shot Tip-Adapter-F 缓存和 R-TPT 测试时提示更新。离线阶段建立图像特征缓存并训练 Tip-Adapter-F；评估阶段为每张图像生成原图视图与伽马、对比度增强视图，用缓存分数引导提示更新及最终分类。

`scripts/prepare_neu_cache.py` 准备划分和特征缓存，`scripts/train_tip_f.py` 训练适配器，`scripts/compare_neu.py` 比较基础方法，`scripts/search_lighting.py` 评估光照条件。共用实现位于 `src/`，R-TPT 和 Tip-Adapter 的固定源码版本位于 `third_party/` 子模块。

## 环境安装

准备 Git、Conda 和支持 CUDA 11.3 的 NVIDIA GPU。在 PowerShell 中克隆仓库并创建环境：

```powershell
git clone --recurse-submodules https://github.com/yicixingderen/R-TPT-Adapter.git
cd R-TPT-Adapter
conda env create -f environment.yml
conda activate rtpt-adapter
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

环境配置固定 Python 3.7.16、PyTorch 1.12.1、torchvision 0.13.1，以及图像处理和 CLIP 分词依赖。[PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)列出对应的 CUDA 11.3 组合。

## 获取代码与子模块

克隆命令同时获取两个固定版本的上游子模块。查看版本：`git submodule status`。

已有本仓库工作目录时，执行 `git submodule update --init --recursive` 获取固定的上游版本。

## 准备数据和权重

从[东北大学 NEU surface defect database](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/)下载 NEU-CLS 图像，并将六类图像分别放在以下目录：

```text
data/NEU-CLS/
├── Cr/   crazing
├── In/   inclusion
├── PS/   pitted surface
├── Pa/   patches
├── RS/   rolled-in scale
└── Sc/   scratches
```

划分文件采用 `data/NEU-CLS/split_NEUCLS.json`，包含 `train`、`val`、`test` 三组，每条记录为 `[相对图像路径, 类别编号, 英文类别名]`。类别编号依次为 `Cr=0`、`In=1`、`PS=2`、`Pa=3`、`RS=4`、`Sc=5`。NEU-CLS 每类有 300 张图像。下节命令会在没有划分文件时先按文件名排序，再按种子 1 生成每类 150/60/90 张的训练、验证和测试划分；已有划分文件会直接使用。

历史实验数字对应当时使用的划分和同一划分生成的缓存。使用这组输入时，按下文的已有缓存步骤执行。

下载 [OpenAI CLIP RN50 权重](https://github.com/openai/CLIP/blob/main/clip/clip.py)：

```powershell
New-Item -ItemType Directory -Force checkpoints | Out-Null
Invoke-WebRequest -Uri 'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt' -OutFile 'checkpoints/RN50.pt'
python scripts/check_setup.py --stage prepare --model-path checkpoints/RN50.pt
```

## 生成缓存

以下命令从训练划分中按种子 1 抽取每类 16 张图像，进行 10 轮缓存增强，并保存验证集和测试集特征：

```powershell
python scripts/prepare_neu_cache.py --create-split --model-path checkpoints/RN50.pt --shots 16 --seed 1 --augment-epochs 10
python scripts/check_setup.py --stage train --model-path checkpoints/RN50.pt --shots 16
```

缓存写入 `outputs/cache/NEU-CLS/`，包括 `keys_16shots.pt`、`values_16shots.pt`、`val_f.pt`、`val_l.pt`、`test_f.pt` 和 `test_l.pt`。使用已有缓存时，跳过缓存生成命令，从 `check_setup.py --stage train` 开始执行。已有缓存可通过各命令的 `--cache-dir` 指定；数据位置通过 `--data-root` 指定。

## 训练

使用 16-shot 缓存初始化 Tip-Adapter-F，训练 20 轮，并按验证集表现选择权重：

```powershell
python scripts/train_tip_f.py --model-path checkpoints/RN50.pt --shots 16 --seed 1 --epochs 20 --output outputs/tip_f_val_selected_16shot_seed1.pt
python scripts/check_setup.py --stage evaluate --model-path checkpoints/RN50.pt --shots 16 --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt
```

训练入口读取 `train` 图像、缓存键值及验证特征；权重文件保存在 `outputs/`。

## 评估

先比较静态 Tip-Adapter-F、R-TPT 及两种融合位置：

```powershell
python scripts/compare_neu.py --model-path checkpoints/RN50.pt --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --rtpt-template tip --val-per-class 60 --test-per-class 90 --seed 0 --output outputs/neu_comparison.json
```

然后以 24 个视图评估伽马＋对比度缓存引导 R-TPT；五种条件为强变暗、强变亮、局部反光、强变暗叠加反光及强变亮叠加反光：

```powershell
python scripts/search_lighting.py --model-path checkpoints/RN50.pt --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --split val --per-class 60 --seed 0 --conditions dark_strong bright_strong glare dark_glare bright_glare --candidates gamma_contrast --output outputs/lighting_val.json
python scripts/search_lighting.py --model-path checkpoints/RN50.pt --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --split test --per-class 90 --seed 0 --conditions dark_strong bright_strong glare dark_glare bright_glare --candidates gamma_contrast --output outputs/lighting_test.json
```

所有入口默认读取 `third_party/R-TPT`、`third_party/Tip-Adapter`、`data/NEU-CLS` 和 `outputs/cache/NEU-CLS`。可用 `--rtpt-root`、`--tip-root`、`--data-root`、`--cache-dir` 指向其他位置。`--model-path` 指向 RN50 权重文件。

## 输出说明

缓存、训练权重和评估 JSON 均位于 `outputs/`。评估 JSON 记录实验参数、各条件准确率和逐图预测；训练权重记录所选轮次及验证集指标。历史实验过程与结果见[历史实验记录](docs/历史实验记录.md)，模型计算过程见[技术实现](docs/伽马对比度缓存引导R-TPT技术实现.md)。

## 来源与引用

- [R-TPT 官方实现与论文](https://github.com/TomSheng21/R-TPT)
- [Tip-Adapter 官方实现与论文](https://github.com/gaopengcuhk/Tip-Adapter)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [东北大学 NEU surface defect database](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/)
