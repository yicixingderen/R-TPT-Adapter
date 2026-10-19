# R-TPT-Adapter: Few-Shot Steel Surface Defect Classification

[中文](README.md) · [Experiment history](docs/历史实验记录.md) · [Technical implementation](docs/伽马对比度缓存引导R-TPT技术实现.md)

## Project overview

This project combines CLIP RN50, a 16-shot Tip-Adapter-F cache, and R-TPT test-time prompt tuning for the six NEU-CLS steel surface defect classes. The offline stage builds an image feature cache and trains Tip-Adapter-F. During evaluation, each image produces an original view and gamma-and-contrast views; cache scores guide prompt tuning and the final classification.

`scripts/prepare_neu_cache.py` prepares the split and feature cache, `scripts/train_tip_f.py` trains the adapter, `scripts/compare_neu.py` compares baseline methods, and `scripts/search_lighting.py` evaluates lighting conditions. Shared code lives in `src/`. The pinned R-TPT and Tip-Adapter sources are Git submodules in `third_party/`.

## Environment setup

Prepare Git, Conda, and an NVIDIA GPU with CUDA 11.3 support. In PowerShell, clone the repository and create the environment:

```powershell
git clone --recurse-submodules https://github.com/yicixingderen/R-TPT-Adapter.git
cd R-TPT-Adapter
conda env create -f environment.yml
conda activate rtpt-adapter
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The environment pins Python 3.7.16, PyTorch 1.12.1, torchvision 0.13.1, and the image-processing and CLIP-tokenizer dependencies. The [PyTorch previous-versions guide](https://pytorch.org/get-started/previous-versions/) lists the matching CUDA 11.3 combination.

## Get the code and submodules

The clone command also fetches both pinned upstream submodules. Check their versions with `git submodule status`.

In an existing checkout, run `git submodule update --init --recursive` to obtain the pinned upstream versions.

## Prepare data and weights

Download the NEU-CLS images from the [Northeastern University NEU surface defect database](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/), then place the six classes in these directories:

```text
data/NEU-CLS/
├── Cr/   crazing
├── In/   inclusion
├── PS/   pitted surface
├── Pa/   patches
├── RS/   rolled-in scale
└── Sc/   scratches
```

The split file is `data/NEU-CLS/split_NEUCLS.json`. It contains `train`, `val`, and `test` arrays, each with entries shaped `[relative image path, class index, English class name]`. Class indices are `Cr=0`, `In=1`, `PS=2`, `Pa=3`, `RS=4`, and `Sc=5`. NEU-CLS has 300 images per class. When the split file is absent, the next section sorts filenames and creates a 150/60/90 train/validation/test split per class with seed 1; an existing split is read directly.

The recorded experiment numbers use their original split and a cache built from that split. Follow the existing-cache steps below when using those inputs.

Download the [OpenAI CLIP RN50 weights](https://github.com/openai/CLIP/blob/main/clip/clip.py):

```powershell
New-Item -ItemType Directory -Force checkpoints | Out-Null
Invoke-WebRequest -Uri 'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt' -OutFile 'checkpoints/RN50.pt'
python scripts/check_setup.py --stage prepare --model-path checkpoints/RN50.pt
```

## Build the cache

The following commands sample 16 training images per class with seed 1, run 10 cache-augmentation passes, and save validation and test features:

```powershell
python scripts/prepare_neu_cache.py --create-split --model-path checkpoints/RN50.pt --shots 16 --seed 1 --augment-epochs 10
python scripts/check_setup.py --stage train --model-path checkpoints/RN50.pt --shots 16
```

The cache is written to `outputs/cache/NEU-CLS/`: `keys_16shots.pt`, `values_16shots.pt`, `val_f.pt`, `val_l.pt`, `test_f.pt`, and `test_l.pt`. When using an existing cache, skip the cache-generation command and start with `check_setup.py --stage train`. Use `--cache-dir` to point every command to an existing cache and `--data-root` to select another data location.

## Train

Initialize Tip-Adapter-F from the 16-shot cache, train for 20 epochs, and select weights on the validation set:

```powershell
python scripts/train_tip_f.py --model-path checkpoints/RN50.pt --shots 16 --seed 1 --epochs 20 --output outputs/tip_f_val_selected_16shot_seed1.pt
python scripts/check_setup.py --stage evaluate --model-path checkpoints/RN50.pt --shots 16 --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt
```

The training entry point reads `train` images, cache keys and values, and validation features. The selected weights are saved in `outputs/`.

## Evaluate

Compare static Tip-Adapter-F, R-TPT, and two fusion placements:

```powershell
python scripts/compare_neu.py --model-path checkpoints/RN50.pt --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --rtpt-template tip --val-per-class 60 --test-per-class 90 --seed 0 --output outputs/neu_comparison.json
```

Then evaluate gamma-and-contrast cache-guided R-TPT with 24 views. The five conditions are strong darkening, strong brightening, local glare, strong darkening with glare, and strong brightening with glare:

```powershell
python scripts/search_lighting.py --model-path checkpoints/RN50.pt --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --split val --per-class 60 --seed 0 --conditions dark_strong bright_strong glare dark_glare bright_glare --candidates gamma_contrast --output outputs/lighting_val.json
python scripts/search_lighting.py --model-path checkpoints/RN50.pt --cache-weights outputs/tip_f_val_selected_16shot_seed1.pt --split test --per-class 90 --seed 0 --conditions dark_strong bright_strong glare dark_glare bright_glare --candidates gamma_contrast --output outputs/lighting_test.json
```

All entry points read `third_party/R-TPT`, `third_party/Tip-Adapter`, `data/NEU-CLS`, and `outputs/cache/NEU-CLS` by default. Use `--rtpt-root`, `--tip-root`, `--data-root`, and `--cache-dir` for other locations. `--model-path` selects the RN50 weight file.

## Outputs

Caches, trained weights, and evaluation JSON files are stored in `outputs/`. Evaluation JSON records experiment parameters, accuracy for each condition, and per-image predictions. The weight file records the selected epoch and validation metrics. See the [experiment history](docs/历史实验记录.md) for recorded runs and the [technical implementation](docs/伽马对比度缓存引导R-TPT技术实现.md) for the model computation.

## Sources and citations

- [R-TPT official implementation and paper](https://github.com/TomSheng21/R-TPT)
- [Tip-Adapter official implementation and paper](https://github.com/gaopengcuhk/Tip-Adapter)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Northeastern University NEU surface defect database](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/)
