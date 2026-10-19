"""Train a Tip-Adapter-F cache with validation-based epoch selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_split
from src.experiment import (choose_training_samples, image_path, load_tip_clip,
                            seed_all)
from src.paths import add_path_arguments, model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser, rtpt=False)
    parser.add_argument("--dataset", default="NEU-CLS")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--backbone", default="RN50")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/tip_f_val_selected_16shot_seed1.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    dataset_root = args.data_root
    cache_root = args.cache_dir
    split = load_split(dataset_root)
    train_entries = choose_training_samples(split["train"], args.shots, args.seed)
    labels = torch.tensor([entry[1] for entry in train_entries], device="cuda")
    keys = torch.load(cache_root / f"keys_{args.shots}shots.pt", map_location="cpu").float().cuda()
    values = torch.load(cache_root / f"values_{args.shots}shots.pt", map_location="cpu").float().cuda()
    val_features = torch.load(cache_root / "val_f.pt", map_location="cpu").float().cuda()
    val_labels = torch.load(cache_root / "val_l.pt", map_location="cpu").cuda()
    assert torch.equal(labels, values.argmax(1))

    tip_clip = load_tip_clip(args.tip_root)
    clip_model, _ = tip_clip.load(model_name(args), device="cuda")
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
