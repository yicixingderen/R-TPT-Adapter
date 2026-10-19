"""Shared sampling, cache, and prompt-tuning operations."""

from __future__ import annotations

import importlib.util
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def choose_training_samples(entries: list, shots: int, seed: int) -> list:
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry[1]].append(entry)
    rng = random.Random(seed)
    selected = []
    for label in sorted(grouped):
        selected.extend(rng.sample(grouped[label], shots))
    return selected


def image_path(dataset_root: Path, entry: list) -> Path:
    relative = entry[0].lstrip("\\/").replace("\\", "/")
    return dataset_root / relative


def load_tip_clip(tip_root: Path):
    """Load Tip-Adapter's CLIP package without colliding with R-TPT's clip."""
    clip_path = tip_root / "clip" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "tip_clip", clip_path, submodule_search_locations=[str(clip_path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["tip_clip"] = module
    spec.loader.exec_module(module)
    return module


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


def set_tip_prompt(model: torch.nn.Module, classnames: list,
                   clip_module, token_embedding: torch.nn.Module) -> None:
    """Insert Tip-Adapter's fixed class suffix after tunable context."""
    learner = model.prompt_learner
    prompts = [f"a photo of {name}, a type of steel surface defect." for name in classnames]
    tokenized = torch.cat([clip_module.tokenize(prompt) for prompt in prompts])
    with torch.no_grad():
        embeddings = token_embedding(tokenized).to(device=learner.ctx.device,
                                               dtype=learner.dtype)
        learner.token_prefix = embeddings[:, :1, :]
        learner.token_suffix = embeddings[:, 1 + learner.n_ctx:, :]
        learner.tokenized_prompts = tokenized.to(learner.ctx.device)


def tune(model: torch.nn.Module, images: torch.Tensor,
         optimizer: torch.optim.Optimizer, initial_state: dict, steps: int,
         selection_p: float, cache: torch.Tensor | None = None,
         alpha: float = 0.0) -> None:
    """Reset the prompt and adapt it for one image's views."""
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
