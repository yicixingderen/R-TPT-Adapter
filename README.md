# R-TPT × Tip-Adapter 探索仓库

本仓库保存两种方法的结合实验代码。原始 `R-TPT` 与 `Tip-Adapter` 保持为独立仓库；数据、缓存与预训练权重仍从各自路径读取，不复制进本仓库。

## 环境与输入

- Python 3.7、PyTorch 1.12.1+cu113、torchvision 0.13.1+cu113；本机可用环境为 `conda run -n tip_adapter`。
- R-TPT 仓库须能运行其 `rtpt.py`，RN50 权重位于 CLIP 默认缓存目录。
- Tip-Adapter 仓库须有 `DATA/NEU-CLS/split_NEUCLS.json` 和 `caches/NEU-CLS/` 中的 16-shot 缓存与验证、测试特征。
- 所有源仓库与数据路径都通过命令行指定。

## 运行

先执行 R-TPT 的单样本 GPU 冒烟测试：

```powershell
conda run -n tip_adapter python rtpt.py <Tip-Adapter-DATA> --test_sets Caltech101 -a RN50 -b 24 --gpu 0 --ctx_init a_photo_of_a -p 1 --eps 0 --max_samples 1 -j 0 --output_dir output_results/smoke
```

上述命令在 **R-TPT 仓库** 执行。随后在本仓库执行完整 NEU-CLS 对比：

```powershell
conda run -n tip_adapter python scripts/compare_neu.py --rtpt-root <R-TPT仓库> --tip-root <Tip-Adapter仓库> --rtpt-template tip --val-per-class 60 --test-per-class 90 --output outputs/neu_16shot_full_tip_prompt_seed0.json
```

可降低 `--val-per-class`、`--test-per-class` 做快速探索。`outputs/` 不纳入 Git。

## 比较方法

| 名称 | 计算方法 |
| --- | --- |
| `clip` | Tip-Adapter 文本模板下的零样本 CLIP |
| `tip_adapter` | 原始 CLIP logits 加缓存检索 logits；`beta`、`alpha` 由完整验证特征网格搜索 |
| `r_tpt` | 单图多视图、低熵视图选择、提示更新、图像相似度加权输出 |
| `r_tpt_plus_tip` | 独立提示更新后，将 Tip 缓存 logits 融合到各视图输出 |
| `cache_guided_r_tpt` | 以提示 logits 加缓存 logits 的熵为目标更新提示，再融合输出 |

每张测试图像都重置提示参数和优化器状态。所有方法用同一组图像及增强视图；`seed=0`，RN50，16-shot 缓存，24 视图，1 次更新。先在验证集选方法，再查看测试集结果。默认 `--rtpt-template tip` 使 R-TPT 的初始提示与 Tip-Adapter 一致：共享的三个可学习上下文词初始化为 `a photo of`，类别名后固定追加 `, a type of steel surface defect.`。程序会输出两条文本分支初始化特征的最小余弦相似度。`--rtpt-template original` 可重现先前的 `a_photo_of_a` 提示。

本项目沿用 Tip-Adapter 工作区现有的数据划分及特征缓存。要重建缓存，请在原 Tip-Adapter 仓库按其配置运行；缓存与拆分文件不在本仓库版本控制中。

## 完整 NEU-CLS 结果

先前的 **原始 R-TPT 提示** 实验中，在既有 16-shot 缓存、360 张验证图像和 540 张测试图像上，`cache_guided_r_tpt` 为 91.11% / 88.89%（480/540），静态 `tip_adapter` 为 90.56% / 88.52%（478/540）。两条文本分支模板不同，因此这组数字不作为统一提示后的结论。原始结果在本地 `outputs/neu_16shot_full_seed0.json`。

**统一为 Tip-Adapter 提示** 后，初始化的两条文本分支特征最小余弦相似度为 1.000000。完整结果如下，方法按验证集准确率选择：

| 方法 | 验证集 | 测试集 |
| --- | ---: | ---: |
| 静态 Tip-Adapter | **90.56%** | **88.52%（478/540）** |
| 单独 R-TPT | 31.94% | 31.67%（171/540） |
| R-TPT 更新后融合缓存 | 86.94% | 84.44%（456/540） |
| 缓存引导 R-TPT | 87.78% | 86.11%（465/540） |

统一模板后，验证集选中静态 Tip-Adapter；缓存引导方案在测试集少正确 13 张。逐图结果保存在本地 `outputs/neu_16shot_full_tip_prompt_seed0.json`。

## Tip-Adapter-F 缓存实验

原 Tip-Adapter 的 `main.py` 用测试集准确率保存最佳 F 轮次。为保持测试集独立，本仓库从 16-shot 缓存键初始化线性适配器，按原配置训练 20 轮，并只用验证集准确率（平局时用验证损失）选择轮次。训练图像只来自既有训练划分；权重写到忽略的 `outputs/`，不会覆盖原仓库缓存。

```powershell
conda run -n tip_adapter python scripts/train_tip_f.py --tip-root <Tip-Adapter仓库> --output outputs/tip_f_val_selected_16shot_seed1.pt
conda run -n tip_adapter python scripts/compare_neu.py --rtpt-root <R-TPT仓库> --tip-root <Tip-Adapter仓库> --rtpt-template tip --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --val-per-class 60 --test-per-class 90 --output outputs/neu_16shot_full_tip_f_guided_seed0.json
```

`--cache-weights` 用 F 适配器的权重替代原始缓存键，同时在验证集重新搜索 `beta` 和 `alpha`。输出分别包含静态 Tip-F、R-TPT 更新后融合 Tip-F，以及由 Tip-F 缓存分数引导提示更新三条路线。

此次训练按验证集选中第 20 轮；后续网格搜索得到 `beta=0.6382`、`alpha=5.0`。统一提示、RN50、360 张验证图像和 540 张测试图像上的结果为：

| 方法 | 验证集 | 测试集 |
| --- | ---: | ---: |
| 静态 Tip-F | **96.11%** | **93.15%（503/540）** |
| R-TPT 更新后融合 Tip-F | 92.50% | 90.74%（490/540） |
| Tip-F 缓存引导 R-TPT | 94.44% | 92.96%（502/540） |

Tip-F 缓存引导方案比原始缓存引导方案测试准确率高 6.85 个百分点（多正确 37 张），但比静态 Tip-F 少正确 1 张。此次方法选择仍应是静态 Tip-F。该结果只覆盖一个划分、一个 F 训练种子和一个测试时增强种子；`outputs/neu_16shot_full_tip_f_guided_seed0.json` 保存了逐图预测。

## 测试时图像变换搜索

为检查 R-TPT 的 AugMix 是否适合 NEU-CLS，`scripts/search_augmentations.py` 固定了 Tip-F 权重、统一文本提示、24 视图、一次提示更新和缓存系数，只改变原图以外 23 个视图的生成方式：

| 候选 | 增强视图 |
| --- | --- |
| `augmix` | 原 R-TPT 的随机裁剪、水平翻转及 AugMix 操作 |
| `identity` | 与确定性原图视图相同，不施加随机变换 |
| `crop` | 轻随机裁剪，面积比例 0.9–1.0、宽高比 0.95–1.05 |
| `photo` | 固定几何预处理，随机轻亮度/对比度调整 |
| `flip` | 固定几何预处理，随机水平/垂直翻转 |
| `crop_photo` | 轻裁剪及轻亮度/对比度调整 |

```powershell
conda run -n tip_adapter python scripts/search_augmentations.py --rtpt-root <R-TPT仓库> --tip-root <Tip-Adapter仓库> --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --val-per-class 60 --test-per-class 90 --seed 0 --output outputs/neu_augmentation_search_seed0.json
conda run -n tip_adapter python scripts/search_augmentations.py --rtpt-root <R-TPT仓库> --tip-root <Tip-Adapter仓库> --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --val-per-class 60 --seed 1 --skip-test --output outputs/neu_augmentation_search_seed1_val.json
```

| 变换 | 验证集种子 0 | 验证集种子 1 |
| --- | ---: | ---: |
| AugMix | 336/360 | 339/360 |
| 重复原图 | **345/360** | **345/360** |
| 轻裁剪 | 343/360 | 343/360 |
| 轻光照 | 343/360 | 341/360 |
| 翻转 | 342/360 | 341/360 |
| 轻裁剪＋轻光照 | 342/360 | 343/360 |

因此在这六组候选中，验证集选择 `identity`。其独立测试集准确率为 502/540（92.96%），静态 Tip-F 为 503/540（93.15%）。重复原图虽然改善了相对于 AugMix 的验证表现，但测试时提示更新仍未超过静态 Tip-F。AugMix 增强视图与原图的平均图像特征余弦相似度为 0.894，其他轻变换约为 0.983–0.993；这与强变换可能扰动缺陷纹理线索的解释一致，但单次数据划分不能证明因果关系。

`compare_neu.py` 也支持 `--augmentation identity` 等候选值，便于直接运行选中的结合方案；默认仍为 `augmix`，以保留早期实验的复现入口。当前最高的独立测试准确率仍来自**静态 Tip-F**，因此它是现阶段的推荐推理方式。

## 合成光照偏移与测试时视图搜索

`scripts/search_lighting.py` 在每张 NEU-CLS 图像上施加确定性的数字光照偏移，再分别评估静态 Tip-F 与 Tip-F 缓存引导 R-TPT。偏移为：原图、亮度乘 0.70、亮度乘 1.30、伽马 1.40、对比度乘 0.70；最后还在测试集上评估亮度和对比度都乘 0.70 的未参与选型组合。这些是软件生成的代理条件，并非真实产线光照样本。这里的“曝光”由数字亮度调整近似，不等同于相机曝光或重拍。

测试时 24 个视图的第一张固定为原图预处理；其余 23 张可分别使用曝光、伽马、对比度的 0.85–1.15 范围随机调整，也可使用两两组合、三项组合或先前的轻裁剪、轻亮度/对比度方案。除视图生成外，统一文本提示、RN50、Tip-F 权重、`beta=0.6382`、`alpha=5.0`、一次提示更新及低熵视图比例 0.1 均固定。

运行格式：

```powershell
conda run -n tip_adapter python scripts/search_lighting.py --rtpt-root <R-TPT仓库> --tip-root <Tip-Adapter仓库> --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --split val --per-class 10 --seed 0 --output outputs/lighting_screen_seed0.json
conda run -n tip_adapter python scripts/search_lighting.py --rtpt-root <R-TPT仓库> --tip-root <Tip-Adapter仓库> --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --split val --per-class 60 --seed 0 --candidates identity crop photo exposure gamma contrast lighting crop_lighting --output outputs/lighting_full_val_seed0.json
conda run -n tip_adapter python scripts/search_lighting.py --rtpt-root <R-TPT仓库> --tip-root <Tip-Adapter仓库> --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --split val --per-class 60 --seed 1 --candidates contrast crop identity gamma --output outputs/lighting_full_val_seed1.json
conda run -n tip_adapter python scripts/search_lighting.py --rtpt-root <R-TPT仓库> --tip-root <Tip-Adapter仓库> --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --split test --per-class 90 --seed 0 --conditions clean dim bright gamma_dark low_contrast dim_low_contrast --candidates contrast --output outputs/lighting_test_contrast_seed0.json
```

测试种子 1 使用最后一条命令并将 `--seed` 改为 `1`、输出名改为 `lighting_test_contrast_seed1.json`。所有输出保存在忽略的 `outputs/`，包含逐图预测。初筛比较 12 个候选；完整验证比较 8 个；种子 1 复核验证集前 4 个。按五个条件的总正确数选择，静态 Tip-F 为 1647/1800；对比度视图在种子 0、1 分别为 1654/1800、1651/1800，轻裁剪分别为 1653/1800、1650/1800。对比度视图以很小的差距胜出。

| 测试条件 | 静态 Tip-F | 对比度视图 R-TPT，种子 0 | 种子 1 |
| --- | ---: | ---: | ---: |
| 干净 | **503/540** | 500/540 | 500/540 |
| 变暗 | 467/540 | **471/540** | **471/540** |
| 变亮 | 492/540 | **496/540** | **495/540** |
| 伽马变暗 | 481/540 | **483/540** | 481/540 |
| 低对比度 | 492/540 | **494/540** | **493/540** |
| 变暗＋低对比度（未参与选型） | **415/540** | 413/540 | 413/540 |

六个条件的测试总正确数为静态 Tip-F 2850/3240、对比度视图 R-TPT 2857/3240 和 2853/3240。它在四个单因素光照偏移上有小幅收益，但干净图像和未参与选型的复合偏移更差；同一批图像在多个条件下重复评估，总计数不能视为 3240 张独立图像。当前结果不足以证明真实工业光照下的提升，干净图像仍建议静态 Tip-F。
