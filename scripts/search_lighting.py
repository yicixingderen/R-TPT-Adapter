"""Compare test-time lighting views on clean and shifted NEU-CLS images."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision import transforms
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.augmentations import build_augmenter
from compare_neu import choose_samples, select_hyperparameters, set_tip_prompt
from search_augmentations import CANDIDATES, evaluate, seed_all


CONDITIONS = ("clean", "dim", "bright", "gamma_dark", "low_contrast",
              "dim_low_contrast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtpt-root", type=Path, required=True)
    parser.add_argument("--tip-root", type=Path, required=True)
    parser.add_argument("--cache-weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--backbone", default="RN50")
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=list(CONDITIONS[:5]))
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES,
                        default=["identity", "crop", "photo", "crop_photo",
                                 "exposure", "gamma", "contrast",
                                 "exposure_gamma", "exposure_contrast",
                                 "gamma_contrast", "lighting", "crop_lighting"])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def condition_transform(name: str) -> Callable[[Image.Image], Image.Image]:
    """Return a deterministic digital proxy for a lighting change."""
    if name == "clean":
        return lambda image: image
    if name == "dim":
        return lambda image: ImageEnhance.Brightness(image).enhance(0.70)
    if name == "bright":
        return lambda image: ImageEnhance.Brightness(image).enhance(1.30)
    if name == "gamma_dark":
        return lambda image: TF.adjust_gamma(image, 1.40)
    if name == "low_contrast":
        return lambda image: ImageEnhance.Contrast(image).enhance(0.70)
    if name == "dim_low_contrast":
        return lambda image: ImageEnhance.Contrast(
            ImageEnhance.Brightness(image).enhance(0.70)).enhance(0.70)
    raise ValueError(name)


def summarize(results: dict, candidates: list[str]) -> dict:
    summary = {}
    for candidate in candidates:
        accuracies = [condition[candidate]["accuracy"]["tip_f_guided_r_tpt"]
                      for condition in results.values() if candidate in condition]
        if accuracies:
            summary[candidate] = {"mean_accuracy": sum(accuracies) / len(accuracies),
                                  "worst_accuracy": min(accuracies),
                                  "conditions_completed": len(accuracies)}
    return summary


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.rtpt_root.resolve()))
    import clip
    from clip.custom_clip import get_coop
    from data.datautils import AugMixAugmenter

    seed_all(args.seed)
    dataset_root = args.tip_root / "DATA" / "NEU-CLS"
    cache_root = args.tip_root / "caches" / "NEU-CLS"
    with (dataset_root / "split_NEUCLS.json").open(encoding="utf-8") as source:
        split = json.load(source)
    classnames = [next(entry[2] for entry in split["train"] if entry[1] == label)
                  for label in sorted({entry[1] for entry in split["train"]})]
    checkpoint = torch.load(args.cache_weights, map_location="cpu")
    keys = checkpoint["weight"].T.float().cuda()
    values = torch.load(cache_root / f"values_{args.shots}shots.pt",
                        map_location="cpu").float().cuda()
    val_features = torch.load(cache_root / "val_f.pt", map_location="cpu").float().cuda()
    val_labels = torch.load(cache_root / "val_l.pt", map_location="cpu").cuda()

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
            tokens = tip_clip.tokenize(
                f"a photo of {classname}, a type of steel surface defect.")
            text_features.append(F.normalize(text_model.encode_text(tokens).float(), dim=-1)[0])
        weights = torch.stack(text_features, dim=1).cuda()
    beta, alpha, _ = select_hyperparameters(
        val_features, val_labels, weights, keys, values)
    model = get_coop(args.backbone, classnames, 0, 3, "a_photo_of").cuda().eval()
    set_tip_prompt(model, classnames, clip, text_model.token_embedding)
    del text_model
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("prompt_learner" in name)
    optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=0.005)
    initial_state = optimizer.state_dict()

    original = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224)])

    def augmix_factory(n_views: int):
        return AugMixAugmenter(original, transforms.ToTensor(),
                               n_views=n_views, augmix=True)

    per_class = args.per_class or (60 if args.split == "val" else 90)
    entries = choose_samples(split[args.split], per_class, args.seed)
    results = {}
    report = {"split": args.split, "per_class": per_class, "seed": args.seed,
              "views": args.views, "shots": args.shots,
              "backbone": args.backbone, "cache_epoch": checkpoint["epoch"],
              "beta": beta, "alpha": alpha,
              "conditions": args.conditions, "candidates": args.candidates,
              "results": results, "summary": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        results[condition] = {}
        for candidate in args.candidates:
            seed_all(args.seed)
            transform = build_augmenter(candidate, args.views - 1, augmix_factory)
            name = f"{condition}/{candidate}"
            results[condition][candidate] = evaluate(
                entries, dataset_root, transform, model, optimizer,
                initial_state, weights, keys, values, beta, alpha, name,
                input_transform=condition_transform(condition))
            report["summary"] = summarize(results, args.candidates)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    for candidate, metrics in sorted(report["summary"].items(),
                                     key=lambda item: item[1]["mean_accuracy"], reverse=True):
        print(f"{candidate}: mean={metrics['mean_accuracy']:.2f}%, "
              f"worst={metrics['worst_accuracy']:.2f}%", flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
