"""NEU-CLS split format and cache file names."""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.experiment import image_path


CLASS_NAMES = {
    "Cr": "crazing",
    "In": "inclusion",
    "PS": "pitted surface",
    "Pa": "patches",
    "RS": "rolled-in scale",
    "Sc": "scratches",
}
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}
CACHE_NAMES = ("keys_{shots}shots.pt", "values_{shots}shots.pt",
               "val_f.pt", "val_l.pt", "test_f.pt", "test_l.pt")


def cache_files(cache_dir: Path, shots: int) -> dict[str, Path]:
    """Return the Tip-Adapter cache layout used by every entry point."""
    return {name.format(shots=shots): cache_dir / name.format(shots=shots)
            for name in CACHE_NAMES}


def load_split(data_root: Path, *, check_images: bool = False) -> dict:
    """Read and validate the six-class NEU-CLS split."""
    split_path = data_root / "split_NEUCLS.json"
    with split_path.open(encoding="utf-8") as source:
        split = json.load(source)
    if set(split) != {"train", "val", "test"}:
        raise ValueError(f"Invalid split sections: {split_path}")
    seen = set()
    class_by_label = list(CLASS_NAMES.values())
    code_by_label = list(CLASS_NAMES)
    for section in ("train", "val", "test"):
        for entry in split[section]:
            if not isinstance(entry, list) or len(entry) != 3:
                raise ValueError(f"Invalid {section} entry in {split_path}")
            path, label, classname = entry
            if not isinstance(path, str) or not isinstance(label, int):
                raise ValueError(f"Invalid {section} entry in {split_path}")
            if (label not in range(6) or classname != class_by_label[label]
                    or path.lstrip("\\/").replace("\\", "/").split("/")[0]
                    != code_by_label[label]):
                raise ValueError(f"Unexpected NEU-CLS class in {split_path}")
            relative = path.lstrip("\\/").replace("\\", "/")
            if relative in seen:
                raise ValueError(f"Duplicate image in {split_path}: {relative}")
            seen.add(relative)
            if check_images and not image_path(data_root, entry).is_file():
                raise FileNotFoundError(image_path(data_root, entry))
    return split


def create_split(data_root: Path, seed: int = 1) -> dict:
    """Create a deterministic 50/20/30 split from six class directories."""
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    for label, (code, classname) in enumerate(CLASS_NAMES.items()):
        folder = data_root / code
        images = sorted(path for path in folder.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(images) < 3:
            raise ValueError(f"At least three images are required in {folder}")
        rng.shuffle(images)
        n_train = round(len(images) * 0.5)
        n_val = round(len(images) * 0.2)
        for section, paths in (("train", images[:n_train]),
                               ("val", images[n_train:n_train + n_val]),
                               ("test", images[n_train + n_val:])):
            split[section].extend([[f"{code}/{path.name}", label, classname]
                                   for path in paths])
    return split
