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
