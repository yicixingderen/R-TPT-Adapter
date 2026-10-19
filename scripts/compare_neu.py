"""Compare R-TPT and Tip-Adapter combinations on an existing NEU-CLS split."""

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
from src.augmentations import build_augmenter
from src.data import load_split
from src.experiment import (cache_logits, choose_samples, image_path, load_tip_clip,
                            seed_all, select_hyperparameters, set_tip_prompt, tune)
from src.paths import add_path_arguments, model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser)
    parser.add_argument("--dataset", default="NEU-CLS")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--cache-weights", type=Path,
                        help="validation-selected Tip-Adapter-F checkpoint")
    parser.add_argument("--backbone", default="RN50")
    parser.add_argument("--rtpt-template", choices=["tip", "original"], default="tip",
                        help="text prompt used by the R-TPT branch")
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--augmentation", default="augmix",
                        choices=["augmix", "identity", "crop", "photo", "flip",
                                 "crop_photo"])
    parser.add_argument("--tta-steps", type=int, default=1)
    parser.add_argument("--selection-p", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--val-per-class", type=int, default=5)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/neu_comparison.json"))
    return parser.parse_args()


def evaluate(entries: list, dataset_root: Path, transform, model: torch.nn.Module,
             optimizer: torch.optim.Optimizer, initial_state: dict,
             weights: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
             beta: float, alpha: float, args: argparse.Namespace,
             cache_variant: str) -> dict:
    if cache_variant == "tip_adapter_f":
        methods = ["clip", "tip_adapter_f", "r_tpt", "r_tpt_plus_tip_f",
                   "tip_f_guided_r_tpt"]
    else:
        methods = ["clip", "tip_adapter", "r_tpt", "r_tpt_plus_tip",
                   "cache_guided_r_tpt"]
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

    seed_all(args.seed)
    dataset_root = args.data_root
    cache_root = args.cache_dir
    split = load_split(dataset_root)
    classnames = [next(entry[2] for entry in split["train"] if entry[1] == label)
                  for label in sorted({entry[1] for entry in split["train"]})]

    if args.cache_weights is None:
        keys = torch.load(cache_root / f"keys_{args.shots}shots.pt", map_location="cpu").float().cuda()
        cache_variant = "tip_adapter"
        cache_epoch = None
    else:
        checkpoint = torch.load(args.cache_weights, map_location="cpu")
        keys = checkpoint["weight"].T.float().cuda()
        cache_variant = "tip_adapter_f"
        cache_epoch = checkpoint["epoch"]
    values = torch.load(cache_root / f"values_{args.shots}shots.pt", map_location="cpu").float().cuda()
    val_features = torch.load(cache_root / "val_f.pt", map_location="cpu").float().cuda()
    val_labels = torch.load(cache_root / "val_l.pt", map_location="cpu").cuda()
    test_features = torch.load(cache_root / "test_f.pt", map_location="cpu").float().cuda()
    test_labels = torch.load(cache_root / "test_l.pt", map_location="cpu").cuda()

    tip_clip = load_tip_clip(args.tip_root)
    text_model, _ = tip_clip.load(model_name(args), device="cpu")
    with torch.no_grad():
        text_features = []
        for classname in classnames:
            tokens = tip_clip.tokenize(f"a photo of {classname}, a type of steel surface defect.")
            feature = text_model.encode_text(tokens).float()
            text_features.append(F.normalize(feature, dim=-1).squeeze(0))
        weights = torch.stack(text_features, dim=1).cuda()
    beta, alpha, cached_val_accuracy = select_hyperparameters(
        val_features, val_labels, weights, keys, values)
    print(f"Full cached validation set: beta={beta:.4f}, alpha={alpha:.4f}, "
          f"{cache_variant}={cached_val_accuracy * 100:.2f}%", flush=True)
    with torch.no_grad():
        cached_test_logits = 100.0 * test_features @ weights
        cached_test_logits += alpha * cache_logits(test_features, keys, values, beta)
        cached_test_accuracy = cached_test_logits.argmax(1).eq(test_labels).float().mean().item() * 100
    print(f"Full cached test set: {cache_variant}={cached_test_accuracy:.2f}%", flush=True)

    if args.rtpt_template == "tip":
        model = get_coop(model_name(args), classnames, 0, 3, "a_photo_of").cuda().eval()
        set_tip_prompt(model, classnames, clip, text_model.token_embedding)
        with torch.no_grad():
            prompt_weights = F.normalize(model.get_text_features().float(), dim=-1)
            agreement = F.cosine_similarity(prompt_weights, weights.T, dim=-1).min().item()
        print(f"Initial R-TPT/Tip text feature minimum cosine: {agreement:.6f}", flush=True)
    else:
        model = get_coop(model_name(args), classnames, 0, 4, "a_photo_of_a").cuda().eval()
    del text_model
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("prompt_learner" in name)
    optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=args.lr)
    initial_state = optimizer.state_dict()
    original = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224)])
    def augmix_factory(n_views: int):
        return AugMixAugmenter(original, transforms.ToTensor(),
                               n_views=n_views, augmix=True)
    transform = build_augmenter(args.augmentation, args.views - 1, augmix_factory)

    val_entries = choose_samples(split["val"], args.val_per_class, args.seed)
    test_entries = choose_samples(split["test"], args.test_per_class, args.seed)
    test_indices = [split["test"].index(entry) for entry in test_entries]
    cached_sample_accuracy = cached_test_logits[test_indices].argmax(1).eq(
        test_labels[test_indices]).float().mean().item() * 100
    print(f"Selected cached test subset: {cache_variant}={cached_sample_accuracy:.2f}%", flush=True)
    print("Evaluating sampled validation split", flush=True)
    validation = evaluate(val_entries, dataset_root, transform, model, optimizer,
                          initial_state, weights, keys, values, beta, alpha, args,
                          cache_variant)
    winner = max(validation["accuracy"], key=validation["accuracy"].get)
    print(f"Selected on sampled validation: {winner}", flush=True)
    print("Evaluating sampled test split", flush=True)
    test = evaluate(test_entries, dataset_root, transform, model, optimizer,
                    initial_state, weights, keys, values, beta, alpha, args,
                    cache_variant)
    report = {"dataset": args.dataset, "shots": args.shots, "backbone": args.backbone,
              "cache_variant": cache_variant, "cache_epoch": cache_epoch,
              "rtpt_template": args.rtpt_template,
              "seed": args.seed, "views": args.views,
              "augmentation": args.augmentation, "tta_steps": args.tta_steps,
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
