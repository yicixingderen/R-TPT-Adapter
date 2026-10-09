"""Train a Tip-Adapter-F cache with validation-based epoch selection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from compare_neu import image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tip-root", type=Path, required=True)
    parser.add_argument("--dataset", default="NEU-CLS")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--backbone", default="RN50")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/tip_f_val_selected.pt"))
    return parser.parse_args()


def choose_training_samples(entries: list, shots: int, seed: int) -> list:
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry[1], []).append(entry)
    rng = random.Random(seed)
    selected = []
    for label in sorted(grouped):
        selected.extend(rng.sample(grouped[label], shots))
    return selected


def load_tip_clip(tip_root: Path):
    clip_path = tip_root / "clip" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "tip_clip", clip_path, submodule_search_locations=[str(clip_path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["tip_clip"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_root = args.tip_root / "DATA" / args.dataset
    cache_root = args.tip_root / "caches" / args.dataset
    with (dataset_root / "split_NEUCLS.json").open(encoding="utf-8") as source:
        split = json.load(source)
    train_entries = choose_training_samples(split["train"], args.shots, args.seed)
    labels = torch.tensor([entry[1] for entry in train_entries], device="cuda")
    keys = torch.load(cache_root / f"keys_{args.shots}shots.pt", map_location="cpu").float().cuda()
    values = torch.load(cache_root / f"values_{args.shots}shots.pt", map_location="cpu").float().cuda()
    val_features = torch.load(cache_root / "val_f.pt", map_location="cpu").float().cuda()
    val_labels = torch.load(cache_root / "val_l.pt", map_location="cpu").cuda()
    assert torch.equal(labels, values.argmax(1))

    tip_clip = load_tip_clip(args.tip_root)
    clip_model, _ = tip_clip.load(args.backbone, device="cuda")
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)
    classnames = [next(entry[2] for entry in split["train"] if entry[1] == label)
                  for label in sorted({entry[1] for entry in split["train"]})]
    with torch.no_grad():
        text_features = []
        for classname in classnames:
            tokens = tip_clip.tokenize(
                f"a photo of {classname}, a type of steel surface defect.").cuda()
            text_features.append(F.normalize(clip_model.encode_text(tokens).float(), dim=-1)[0])
        weights = torch.stack(text_features, dim=1)

    adapter = torch.nn.Linear(keys.shape[0], keys.shape[1], bias=False).cuda()
    with torch.no_grad():
        adapter.weight.copy_(keys.T)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001, eps=1e-4)
    batches_per_epoch = (len(train_entries) + args.batch_size - 1) // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * batches_per_epoch)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    best = (-1.0, float("inf"))
    best_weight = None
    best_epoch = None
    for epoch in range(args.epochs):
        order = torch.randperm(len(train_entries)).tolist()
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            views = []
            for index in indices:
                with Image.open(image_path(dataset_root, train_entries[index])) as image:
                    views.append(transform(image.convert("RGB")))
            images = torch.stack(views).cuda()
            batch_labels = labels[indices]
            with torch.no_grad():
                features = F.normalize(clip_model.encode_image(images).float(), dim=-1)
            affinity = adapter(features)
            cache = torch.exp(-(1.0 - affinity)) @ values
            logits = 100.0 * features @ weights + 3.0 * cache
            loss = F.cross_entropy(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            affinity = adapter(val_features)
            val_cache = torch.exp(-(1.0 - affinity)) @ values
            val_logits = 100.0 * val_features @ weights + 3.0 * val_cache
            val_accuracy = val_logits.argmax(1).eq(val_labels).float().mean().item()
            val_loss = F.cross_entropy(val_logits, val_labels).item()
        print(f"epoch={epoch + 1} val_acc={val_accuracy * 100:.2f}% "
              f"val_loss={val_loss:.4f}", flush=True)
        if (val_accuracy, -val_loss) > (best[0], -best[1]):
            best = (val_accuracy, val_loss)
            best_weight = adapter.weight.detach().cpu().clone()
            best_epoch = epoch + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": best_weight, "epoch": best_epoch,
                "val_accuracy": best[0], "val_loss": best[1],
                "shots": args.shots, "backbone": args.backbone, "seed": args.seed}, args.output)
    print(f"Selected epoch {best_epoch}, val accuracy {best[0] * 100:.2f}%", flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
