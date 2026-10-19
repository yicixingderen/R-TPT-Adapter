"""Create a NEU-CLS split and the six tensors used by Tip-Adapter-F."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import cache_files, create_split, load_split
from src.experiment import (choose_training_samples, image_path, load_tip_clip,
                            seed_all)
from src.paths import add_path_arguments, model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser, rtpt=False)
    parser.add_argument("--backbone", default="RN50")
    parser.add_argument("--create-split", action="store_true",
                        help="create split_NEUCLS.json when it does not exist")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--augment-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def encode_entries(entries: list, data_root: Path, transform, model,
                   batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode entries in split order without changing their labels."""
    features = []
    for start in range(0, len(entries), batch_size):
        batch = entries[start:start + batch_size]
        images = []
        for entry in batch:
            with Image.open(image_path(data_root, entry)) as source:
                images.append(transform(source.convert("RGB")))
        tensor = torch.stack(images).cuda()
        with torch.no_grad():
            features.append(model.encode_image(tensor).float().cpu())
    labels = torch.tensor([entry[1] for entry in entries], dtype=torch.long)
    return torch.cat(features), labels


def main() -> None:
    args = parse_args()
    if args.shots < 1 or args.augment_epochs < 1 or args.batch_size < 1:
        raise ValueError("shots, augment-epochs and batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the reference cache")
    split_path = args.data_root / "split_NEUCLS.json"
    if not split_path.exists():
        if not args.create_split:
            raise FileNotFoundError(split_path)
        split = create_split(args.data_root, args.seed)
        split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        print(f"Created {split_path}", flush=True)
    split = load_split(args.data_root, check_images=True)
    files = cache_files(args.cache_dir, args.shots)
    existing = [path for path in files.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Cache files already exist: {existing}")
    train_entries = choose_training_samples(split["train"], args.shots, args.seed)
    seed_all(args.seed)
    tip_clip = load_tip_clip(args.tip_root)
    model, preprocess = tip_clip.load(model_name(args), device="cuda")
    model.eval()
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    epochs = []
    for index in range(args.augment_epochs):
        features, labels = encode_entries(train_entries, args.data_root,
                                          train_transform, model, args.batch_size)
        epochs.append(features)
        print(f"Cache augmentation {index + 1}/{args.augment_epochs}", flush=True)
    keys = torch.stack(epochs).mean(0)
    keys = F.normalize(keys, dim=-1).T.contiguous()
    values = F.one_hot(labels, num_classes=6).float()
    val_features, val_labels = encode_entries(
        split["val"], args.data_root, preprocess, model, args.batch_size)
    test_features, test_labels = encode_entries(
        split["test"], args.data_root, preprocess, model, args.batch_size)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tensors = (keys, values, F.normalize(val_features, dim=-1), val_labels,
               F.normalize(test_features, dim=-1), test_labels)
    for path, tensor in zip(files.values(), tensors):
        torch.save(tensor, path)
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
