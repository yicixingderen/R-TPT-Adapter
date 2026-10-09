"""Select test-time transforms on validation data and evaluate finalists."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.augmentations import build_augmenter
from compare_neu import (cache_logits, choose_samples, image_path,
                         select_hyperparameters, set_tip_prompt, tune)


CANDIDATES = ("augmix", "identity", "crop", "photo", "flip", "crop_photo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtpt-root", type=Path, required=True)
    parser.add_argument("--tip-root", type=Path, required=True)
    parser.add_argument("--cache-weights", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--backbone", default="RN50")
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--val-per-class", type=int, default=60)
    parser.add_argument("--test-per-class", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true",
                        help="evaluate candidates on validation only")
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES,
                        default=list(CANDIDATES))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/augmentation_search.json"))
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(entries: list, dataset_root: Path, transform, model,
             optimizer, initial_state: dict, weights: torch.Tensor,
             keys: torch.Tensor, values: torch.Tensor,
             beta: float, alpha: float, name: str) -> dict:
    correct = {"tip_adapter_f": 0, "tip_f_guided_r_tpt": 0}
    predictions = []
    similarity_total = 0.0
    original_weight_total = 0.0
    for index, entry in enumerate(entries, 1):
        with Image.open(image_path(dataset_root, entry)) as source:
            views = transform(source.convert("RGB"))
        images = torch.stack(views).cuda()
        with torch.no_grad():
            features, _, _ = model.forward_features(images)
            features = features.float()
            cache = cache_logits(features, keys, values, beta)
            similarity = features @ features.T
            similarity_total += similarity[0, 1:].mean().item()
            similarity.fill_diagonal_(-float("inf"))
            neighbors = similarity.topk(min(20, len(images) - 1), dim=-1).values.mean(-1)
            view_weights = F.softmax(neighbors / 0.01, dim=0)
            original_weight_total += view_weights[0].item()
            static_logits = 100.0 * features[0] @ weights + alpha * cache[0]

        tune(model, images, optimizer, initial_state, 1, 0.1, cache, alpha)
        with torch.no_grad():
            guided_logits = view_weights @ (model(images).float() + alpha * cache)
        predicted = {
            "tip_adapter_f": static_logits.argmax().item(),
            "tip_f_guided_r_tpt": guided_logits.argmax().item(),
        }
        for method, label in predicted.items():
            correct[method] += int(label == entry[1])
        predictions.append({"path": entry[0], "label": entry[1], "predicted": predicted})
        if index % 60 == 0 or index == len(entries):
            print(f"{name}: {index}/{len(entries)} guided="
                  f"{100 * correct['tip_f_guided_r_tpt'] / index:.2f}%", flush=True)
    return {"count": len(entries), "correct": correct,
            "accuracy": {method: 100 * count / len(entries)
                         for method, count in correct.items()},
            "mean_original_view_cosine": similarity_total / len(entries),
            "mean_original_view_weight": original_weight_total / len(entries),
            "predictions": predictions}


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
    values = torch.load(cache_root / f"values_{args.shots}shots.pt", map_location="cpu").float().cuda()
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

    validation_entries = choose_samples(split["val"], args.val_per_class, args.seed)
    test_entries = choose_samples(split["test"], args.test_per_class, args.seed)
    validation = {}
    for name in args.candidates:
        seed_all(args.seed)
        transform = build_augmenter(name, args.views - 1, augmix_factory)
        validation[name] = evaluate(validation_entries, dataset_root, transform,
                                    model, optimizer, initial_state, weights,
                                    keys, values, beta, alpha, name)
    selected = max(args.candidates,
                   key=lambda name: validation[name]["accuracy"]["tip_f_guided_r_tpt"])
    print(f"Selected on validation: {selected}", flush=True)
    test = None
    if not args.skip_test:
        seed_all(args.seed)
        test_transform = build_augmenter(selected, args.views - 1, augmix_factory)
        test = evaluate(test_entries, dataset_root, test_transform, model, optimizer,
                        initial_state, weights, keys, values, beta, alpha, selected)
    report = {"candidates": args.candidates, "selected": selected,
              "seed": args.seed, "views": args.views, "shots": args.shots,
              "backbone": args.backbone, "cache_epoch": checkpoint["epoch"],
              "beta": beta, "alpha": alpha,
              "validation": validation, "test": test}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
