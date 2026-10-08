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
conda run -n tip_adapter python scripts/compare_neu.py --rtpt-root <R-TPT仓库> --tip-root <Tip-Adapter仓库> --val-per-class 60 --test-per-class 90 --output outputs/neu_16shot_full_seed0.json
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

每张测试图像都重置提示参数和优化器状态。所有方法用同一组图像及增强视图；`seed=0`，RN50，16-shot 缓存，24 视图，1 次更新。先在验证集选方法，再查看测试集结果。单独 R-TPT 的提示为 `a_photo_of_a`；Tip-Adapter 保持 NEU-CLS 原有文本模板，因此两条零样本文本分支的绝对准确率不能当成同一提示的消融实验。

本项目沿用 Tip-Adapter 工作区现有的数据划分及特征缓存。要重建缓存，请在原 Tip-Adapter 仓库按其配置运行；缓存与拆分文件不在本仓库版本控制中。

## 完整 NEU-CLS 结果

在既有 16-shot 缓存、360 张验证图像和 540 张测试图像上，`cache_guided_r_tpt` 由验证集选中：验证准确率 91.11%，测试准确率 88.89%（480/540）。静态 `tip_adapter` 为 90.56% / 88.52%（478/540）。净增 2 张测试图像，且两者在 72 张测试图像上有相反的正确性，因此这一次运行只能说明该配置有轻微收益，不能说明稳定改进。逐图预测保存在本地 `outputs/neu_16shot_full_seed0.json`。
