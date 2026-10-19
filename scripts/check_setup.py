"""Check files and environment for the NEU-CLS reproduction commands."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASS_NAMES, IMAGE_EXTENSIONS, cache_files, load_split
from src.experiment import load_tip_clip
from src.paths import add_path_arguments


RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser)
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate"),
                        default="evaluate")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--cache-weights", type=Path,
                        default=Path("outputs/tip_f_val_selected_16shot_seed1.pt"))
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def check_version(name: str, actual: str, expected: str) -> None:
    if not actual.startswith(expected):
        raise RuntimeError(f"{name}: expected {expected}, found {actual}")
    print(f"{name}: {actual}")


def check_model(path: Path) -> None:
    require_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != RN50_SHA256:
        raise ValueError(f"RN50 SHA256 mismatch: {path}")
    print(f"RN50 checkpoint: {path}")


def main() -> None:
    args = parse_args()
    import numpy
    import PIL
    import torch
    import torchvision
    import ftfy
    import regex
    import tqdm
    import pkg_resources

    check_version("Python", sys.version.split()[0], "3.7.")
    check_version("PyTorch", torch.__version__, "1.12.1")
    check_version("torchvision", torchvision.__version__, "0.13.1")
    check_version("NumPy", numpy.__version__, "1.21.")
    check_version("Pillow", PIL.__version__, "9.4.")
    print(f"Tokenizer dependencies: ftfy {ftfy.__version__}, regex {regex.__version__}, "
          f"tqdm {tqdm.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    require_file(args.rtpt_root / "clip" / "custom_clip.py")
    require_file(args.tip_root / "clip" / "clip.py")
    sys.path.insert(0, str(args.rtpt_root.resolve()))
    from clip.custom_clip import get_coop
    from data.datautils import AugMixAugmenter
    load_tip_clip(args.tip_root)
    print("Source submodules: ready")
    if args.model_path is None:
        raise ValueError("Pass --model-path checkpoints/RN50.pt")
    check_model(args.model_path)
    if args.stage == "prepare" and not (args.data_root / "split_NEUCLS.json").exists():
        for code in CLASS_NAMES:
            folder = args.data_root / code
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            count = sum(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                        for path in folder.iterdir())
            if count < args.shots * 2:
                raise ValueError(f"Insufficient images for {args.shots} shots: {folder}")
        print("NEU-CLS image directories: ready for split creation")
        return
    split = load_split(args.data_root, check_images=True)
    counts = Counter(entry[1] for entry in split["train"])
    if set(counts) != set(range(6)) or min(counts.values()) < args.shots:
        raise ValueError("Training split has fewer than --shots images in a class")
    print("NEU-CLS split and images: ready")
    if args.stage == "prepare":
        return
    files = cache_files(args.cache_dir, args.shots)
    for path in files.values():
        require_file(path)
    tensors = {name: torch.load(path, map_location="cpu")
               for name, path in files.items()}
    keys = tensors[f"keys_{args.shots}shots.pt"]
    values = tensors[f"values_{args.shots}shots.pt"]
    if keys.ndim != 2 or values.shape != (keys.shape[1], 6):
        raise ValueError("Cache key/value shapes do not match")
    for section in ("val", "test"):
        features = tensors[f"{section}_f.pt"]
        labels = tensors[f"{section}_l.pt"]
        if (features.ndim != 2 or features.shape[1] != keys.shape[0]
                or len(features) != len(labels) or len(labels) != len(split[section])):
            raise ValueError(f"{section} feature/label shapes do not match split")
    print("Cache tensors: ready")
    if args.stage == "train":
        return
    require_file(args.cache_weights)
    checkpoint = torch.load(args.cache_weights, map_location="cpu")
    if checkpoint["weight"].shape != (keys.shape[1], keys.shape[0]):
        raise ValueError("Tip-Adapter-F weight shape does not match cache")
    print("Tip-Adapter-F checkpoint: ready")


if __name__ == "__main__":
    main()
