"""Repository defaults and command-line paths for NEU-CLS experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RTPT_ROOT = PROJECT_ROOT / "third_party" / "R-TPT"
DEFAULT_TIP_ROOT = PROJECT_ROOT / "third_party" / "Tip-Adapter"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "NEU-CLS"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache" / "NEU-CLS"


def add_path_arguments(parser: argparse.ArgumentParser, *, rtpt: bool = True) -> None:
    """Add the paths shared by training and evaluation entry points."""
    if rtpt:
        parser.add_argument("--rtpt-root", type=Path, default=DEFAULT_RTPT_ROOT,
                            help="R-TPT source checkout (default: bundled submodule)")
    parser.add_argument("--tip-root", type=Path, default=DEFAULT_TIP_ROOT,
                        help="Tip-Adapter source checkout (default: bundled submodule)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="NEU-CLS folder containing split_NEUCLS.json")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="directory containing Tip-Adapter cache tensors")
    parser.add_argument("--model-path", type=Path,
                        help="local CLIP RN50 checkpoint; overrides --backbone")


def model_name(args: argparse.Namespace) -> str:
    """Return a checkpoint path or the named CLIP backbone."""
    return str(args.model_path.resolve()) if args.model_path else args.backbone
