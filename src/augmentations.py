"""Candidate test-time view transformations for steel defects."""

from __future__ import annotations

from PIL import Image
from torchvision import transforms


class ViewAugmenter:
    """Keep the original view fixed while sampling transformed views."""

    def __init__(self, transform, n_views: int):
        resize = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
        self.original = transforms.Compose([resize, transforms.CenterCrop(224),
                                            transforms.ToTensor()])
        self.transform = transforms.Compose([transform, transforms.ToTensor()])
        self.n_views = n_views

    def __call__(self, image: Image.Image) -> list:
        return [self.original(image)] + [self.transform(image) for _ in range(self.n_views)]


def build_augmenter(name: str, n_views: int, augmix_augmenter=None):
    """Return a named transformation with the same deterministic first view."""
    if name == "augmix":
        if augmix_augmenter is None:
            raise ValueError("augmix_augmenter is required for the augmix candidate")
        return augmix_augmenter(n_views)

    resize = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
    crop = transforms.RandomResizedCrop(
        224, scale=(0.9, 1.0), ratio=(0.95, 1.05),
        interpolation=transforms.InterpolationMode.BICUBIC)
    photo = transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.8)
    flips = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5)])
    candidates = {
        "identity": transforms.Compose([resize, transforms.CenterCrop(224)]),
        "crop": crop,
        "photo": transforms.Compose([resize, transforms.CenterCrop(224), photo]),
        "flip": transforms.Compose([resize, transforms.CenterCrop(224), flips]),
        "crop_photo": transforms.Compose([crop, photo]),
    }
    return ViewAugmenter(candidates[name], n_views)
