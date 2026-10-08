"""Compare R-TPT and Tip-Adapter combinations on an existing NEU-CLS split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtpt-root", type=Path, required=True)
    parser.add_argument("--tip-root", type=Path, required=True)
    parser.add_argument("--dataset", default="NEU-CLS")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--backbone", default="RN50")
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--tta-steps", type=int, default=1)
    parser.add_argument("--selection-p", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--val-per-class", type=int, default=5)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/neu_comparison.json"))
    return parser.parse_args()


def choose_samples(entries: list, per_class: int, seed: int) -> list:
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry[1]].append(entry)
    rng = random.Random(seed)
    selected = []
    for label in sorted(grouped):
        rng.shuffle(grouped[label])
        selected.extend(grouped[label][:per_class])
    return selected


def cache_logits(features: torch.Tensor, keys: torch.Tensor,
                 values: torch.Tensor, beta: float) -> torch.Tensor:
    affinity = features.float() @ keys
    return torch.exp(-beta * (1.0 - affinity)) @ values


def select_hyperparameters(features: torch.Tensor, labels: torch.Tensor,
                           weights: torch.Tensor, keys: torch.Tensor,
                           values: torch.Tensor) -> tuple[float, float, float]:
    clip_logits = 100.0 * features @ weights
    best = (-1.0, 1.0, 1.0)
    for beta in [0.1 + i * 11.9 / 199 for i in range(200)]:
        cache = cache_logits(features, keys, values, beta)
        for alpha in [0.1 + i * 4.9 / 19 for i in range(20)]:
            accuracy = (clip_logits + alpha * cache).argmax(1).eq(labels).float().mean().item()
            if accuracy > best[0]:
                best = (accuracy, beta, alpha)
    return best[1], best[2], best[0]


def image_path(dataset_root: Path, entry: list) -> Path:
    relative = entry[0].lstrip("\\/").replace("\\", "/")
    return dataset_root / relative


def tune(model: torch.nn.Module, images: torch.Tensor, optimizer: torch.optim.Optimizer,
         initial_state: dict, steps: int, selection_p: float,
         cache: torch.Tensor | None = None, alpha: float = 0.0) -> None:
    with torch.no_grad():
        model.reset()
    optimizer.load_state_dict(initial_state)
    selected = None
    for _ in range(steps):
        logits = model(images)
        if cache is not None:
            logits = logits + alpha * cache
        entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        if selected is None:
            selected = entropy.argsort()[:max(1, int(len(entropy) * selection_p))]
        loss = entropy[selected].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(entries: list, dataset_root: Path, transform, model: torch.nn.Module,
             optimizer: torch.optim.Optimizer, initial_state: dict,
             weights: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
             beta: float, alpha: float, args: argparse.Namespace) -> dict:
    methods = ["clip", "tip_adapter", "r_tpt", "r_tpt_plus_tip", "cache_guided_r_tpt"]
    correct = {method: 0 for method in methods}
    predictions = []
    for index, entry in enumerate(entries, 1):
        with Image.open(image_path(dataset_root, entry)) as source:
            views = transform(source.convert("RGB"))
        images = torch.stack(views).cuda()
        target = entry[1]
        with torch.no_grad():
            features, _, _ = model.forward_features(images)
            features = features.float()
            cache = cache_logits(features, keys, values, beta)
            similarity = features @ features.T
            similarity.fill_diagonal_(-float("inf"))
            neighbors = similarity.topk(min(20, len(images) - 1), dim=-1).values.mean(-1)
            view_weights = F.softmax(neighbors / 0.01, dim=0)
            clip = 100.0 * features[0] @ weights
            tip = clip + alpha * cache[0]

        tune(model, images, optimizer, initial_state, args.tta_steps, args.selection_p)
        with torch.no_grad():
            prompt_logits = model(images).float()
            r_tpt = view_weights @ prompt_logits
            fused = view_weights @ (prompt_logits + alpha * cache)

        tune(model, images, optimizer, initial_state, args.tta_steps,
             args.selection_p, cache, alpha)
        with torch.no_grad():
            guided = view_weights @ (model(images).float() + alpha * cache)

        scores = dict(zip(methods, [clip, tip, r_tpt, fused, guided]))
        predicted = {name: score.argmax().item() for name, score in scores.items()}
        for name, label in predicted.items():
            correct[name] += int(label == target)
        predictions.append({"path": entry[0], "label": target, "predicted": predicted})
        if index % 10 == 0 or index == len(entries):
            print(f"{index}/{len(entries)}: " + ", ".join(
                f"{name}={100 * count / index:.1f}%" for name, count in correct.items()), flush=True)
    return {"count": len(entries), "accuracy": {
        name: 100 * count / len(entries) for name, count in correct.items()},
        "predictions": predictions}


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.rtpt_root.resolve()))
    import clip
    from clip.custom_clip import get_coop
    from data.datautils import AugMixAugmenter

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset_root = args.tip_root / "DATA" / args.dataset
    cache_root = args.tip_root / "caches" / args.dataset
    with (dataset_root / "split_NEUCLS.json").open(encoding="utf-8") as source:
        split = json.load(source)
    classnames = [next(entry[2] for entry in split["train"] if entry[1] == label)
                  for label in sorted({entry[1] for entry in split["train"]})]

    keys = torch.load(cache_root / f"keys_{args.shots}shots.pt", map_location="cpu").float().cuda()
    values = torch.load(cache_root / f"values_{args.shots}shots.pt", map_location="cpu").float().cuda()
    val_features = torch.load(cache_root / "val_f.pt", map_location="cpu").float().cuda()
    val_labels = torch.load(cache_root / "val_l.pt", map_location="cpu").cuda()
    test_features = torch.load(cache_root / "test_f.pt", map_location="cpu").float().cuda()
    test_labels = torch.load(cache_root / "test_l.pt", map_location="cpu").cuda()

    tip_clip_path = args.tip_root / "clip" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "tip_clip", tip_clip_path, submodule_search_locations=[str(tip_clip_path.parent)])
    tip_clip = importlib.util.module_from_spec(spec)
    sys.modules["tip_clip"] = tip_clip
    spec.loader.exec_module(tip_clip)
    text_model, _ = tip_clip.load(args.backbone, device="cpu")
    with torch.no_grad():
        text_features = []
        for classname in classnames:
            tokens = tip_clip.tokenize(f"a photo of {classname}, a type of steel surface defect.")
            feature = text_model.encode_text(tokens).float()
            text_features.append(F.normalize(feature, dim=-1).squeeze(0))
        weights = torch.stack(text_features, dim=1).cuda()
    del text_model

    beta, alpha, cached_val_accuracy = select_hyperparameters(
        val_features, val_labels, weights, keys, values)
    print(f"Full cached validation set: beta={beta:.4f}, alpha={alpha:.4f}, "
          f"Tip-Adapter={cached_val_accuracy * 100:.2f}%", flush=True)
    with torch.no_grad():
        cached_test_logits = 100.0 * test_features @ weights
        cached_test_logits += alpha * cache_logits(test_features, keys, values, beta)
        cached_test_accuracy = cached_test_logits.argmax(1).eq(test_labels).float().mean().item() * 100
    print(f"Full cached test set: Tip-Adapter={cached_test_accuracy:.2f}%", flush=True)

    model = get_coop(args.backbone, classnames, 0, 4, "a_photo_of_a").cuda().eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("prompt_learner" in name)
    optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=args.lr)
    initial_state = optimizer.state_dict()
    transform = AugMixAugmenter(
        transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.CenterCrop(224)]),
        transforms.ToTensor(), n_views=args.views - 1, augmix=True)

    val_entries = choose_samples(split["val"], args.val_per_class, args.seed)
    test_entries = choose_samples(split["test"], args.test_per_class, args.seed)
    test_indices = [split["test"].index(entry) for entry in test_entries]
    cached_sample_accuracy = cached_test_logits[test_indices].argmax(1).eq(
        test_labels[test_indices]).float().mean().item() * 100
    print(f"Selected cached test subset: Tip-Adapter={cached_sample_accuracy:.2f}%", flush=True)
    print("Evaluating sampled validation split", flush=True)
    validation = evaluate(val_entries, dataset_root, transform, model, optimizer,
                          initial_state, weights, keys, values, beta, alpha, args)
    winner = max(validation["accuracy"], key=validation["accuracy"].get)
    print(f"Selected on sampled validation: {winner}", flush=True)
    print("Evaluating sampled test split", flush=True)
    test = evaluate(test_entries, dataset_root, transform, model, optimizer,
                    initial_state, weights, keys, values, beta, alpha, args)
    report = {"dataset": args.dataset, "shots": args.shots, "backbone": args.backbone,
              "seed": args.seed, "views": args.views, "tta_steps": args.tta_steps,
              "selection_p": args.selection_p, "lr": args.lr, "beta": beta,
              "alpha": alpha, "cached_val_accuracy": cached_val_accuracy * 100,
              "cached_test_accuracy": cached_test_accuracy,
              "cached_sample_test_accuracy": cached_sample_accuracy,
              "selected_method": winner, "validation": validation, "test": test}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
