#!/usr/bin/env python3
"""Prefetch translation models used by fill_missing_translations.py for offline use."""

from __future__ import annotations

import argparse
from typing import Iterable, List

from huggingface_hub import snapshot_download

# Default model list mirrors pick_models() in fill_missing_translations.py
DEFAULT_MODELS = [
    # Language-specific Marian models
    "Helsinki-NLP/opus-mt-fr-en",
    "Helsinki-NLP/opus-mt-es-en",
    "Helsinki-NLP/opus-mt-pt-en",
    "Helsinki-NLP/opus-mt-it-en",
    "Helsinki-NLP/opus-mt-de-en",
    "Helsinki-NLP/opus-mt-lv-en",
    "Helsinki-NLP/opus-mt-ga-en",  # Irish Gaelic
    "Helsinki-NLP/opus-mt-gd-en",  # Scottish Gaelic
    "Helsinki-NLP/opus-mt-ru-en",
    # Fallback multi-lingual Marian models
    "Helsinki-NLP/opus-mt-mul-en",
    "Helsinki-NLP/opus-mt-mul-mul",
]


def download_all(models: Iterable[str], cache_dir: str | None, local_dir: str | None) -> None:
    for name in models:
        print(f"Downloading {name} ...", flush=True)
        snapshot_download(
            repo_id=name,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_dir_use_symlinks=True,
            ignore_patterns=["*.msgpack", "*.h5"],  # keep downloads lean
        )
    print("Done. Models are cached locally.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prefetch translation models for offline/fast use.")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Specific models to download. Defaults to the curated set used by fill_missing_translations.py.",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HF cache dir (defaults to HF_HOME/TRANSFORMERS_CACHE).",
    )
    ap.add_argument(
        "--local-dir",
        default=None,
        help="Optional local directory to store the models (instead of the HF cache).",
    )
    args = ap.parse_args()

    models: List[str] = args.models if args.models else list(DEFAULT_MODELS)
    download_all(models, cache_dir=args.cache_dir, local_dir=args.local_dir)


if __name__ == "__main__":
    main()
