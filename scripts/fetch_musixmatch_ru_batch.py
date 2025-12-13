#!/usr/bin/env python3
"""
Fetch a larger batch of Musixmatch songs (default 40) for one language, translate
them with Ollama, and write app-ready JSONs (with grammar) in static/data.

This is a thin wrapper around fetch_musixmatch_songs.py tuned for bulk Russian
ingest, but you can point it at any language code Musixmatch supports.

Env:
  MUSIXMATCH_API_KEY (required)
  OLLAMA_BASE_URL (default http://localhost:11434)
  OLLAMA_MODEL (default llama3)

Example (40 Russian songs in one go):
  MUSIXMATCH_API_KEY=... OLLAMA_BASE_URL=http://localhost:11434 \\
  python scripts/fetch_musixmatch_ru_batch.py --language ru --count 40
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv

# Reuse the battle-tested helpers from fetch_musixmatch_songs
from fetch_musixmatch_songs import (  # type: ignore
    fetch_for_language,
    get_env,
    load_existing_catalog,
)

# Borrow translation/grammar fill logic so we can guarantee every line is populated
from fill_missing_translations import (  # type: ignore
    process_file as process_translations,
)
from fill_missing_grammar import (  # type: ignore
    process_file as process_grammar,
    pick_models as pick_grammar_models,
    load_generator as load_grammar_generator,
)
from fill_missing_translations import norm_lang, pick_models, load_translator  # type: ignore


def main():
    ap = argparse.ArgumentParser(description="Bulk Musixmatch fetch+translate+grammar for one language")
    ap.add_argument("--language", default="ru", help="Musixmatch language code (default: ru)")
    ap.add_argument("--count", type=int, default=40, help="How many songs to fetch (default: 40)")
    ap.add_argument("--skip-annotation", action="store_true", help="Fetch raw lyrics/meta only (no translation/grammar/JSON)")
    ap.add_argument("--verbose", action="store_true", help="Print skip reasons")
    ap.add_argument("--fill-missing", action="store_true", default=True, help="After fetching, force-translate and fill grammar for ALL lines using fill_missing_* logic (default: on)")
    ap.add_argument("--translation-engine", choices=["google", "hf"], default="google", help="Engine for the fill step (default: google API)")
    ap.add_argument("--google-api-key", default=None, help="Google Translate API key (else GOOGLE_API_KEY env)")
    ap.add_argument("--grammar-engine", choices=["ollama", "hf"], default="hf", help="Engine for grammar fill (default: HF local)")
    ap.add_argument("--grammar-model", default=None, help="Override grammar model name (HF) or Ollama model name (ollama)")
    ap.add_argument("--translation-model", default=None, help="Override HF translation model name")
    ap.add_argument("--batch-size", type=int, default=12, help="Batch size for fill steps")
    args = ap.parse_args()

    load_dotenv()
    api_key = get_env("MUSIXMATCH_API_KEY", required=True)
    if args.skip_annotation:
        ollama_url = ""
        ollama_model = ""
    else:
        ollama_url = get_env("OLLAMA_BASE_URL", "") or ("http://" + get_env("OLLAMA_HOST", "localhost:11434"))
        ollama_model = get_env("OLLAMA_MODEL", "llama3")

    lang_code = args.language.strip()
    if not lang_code:
        raise SystemExit("language code is required")

    seen_pairs = load_existing_catalog()
    print(f"=== Fetching {args.count} songs for language {lang_code} ===")
    saved = fetch_for_language(
        display_lang=lang_code,
        lang_codes=[lang_code],
        api_key=api_key,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        seen_pairs=seen_pairs,
        per_count=args.count,
        skip_annotation=args.skip_annotation,
        verbose=args.verbose,
    )
    print(f"Done. New items: {len(saved)}")
    for p in saved:
        try:
            rel = p.relative_to(Path(__file__).resolve().parents[1])
        except Exception:
            rel = p
        print(f"  - {rel}")

    if args.fill_missing:
        google_key = args.google_api_key or os.getenv("GOOGLE_API_KEY", "")
        if trans_engine == "google" and not google_key:
            raise SystemExit("GOOGLE_API_KEY is required for --fill-missing with translation-engine=google")
        trans_engine = args.translation_engine
        gram_engine = args.grammar_engine
        new_jsons = [Path(p) for p in saved if str(p).endswith(".json")]
        if not new_jsons:
            print("No JSONs to fill; skipping fill_missing steps.")
            return
        print(f"\n=== Filling translations for {len(new_jsons)} files (engine={trans_engine}) ===")
        trans_cache: Dict[str, object] = {}
        for path in new_jsons:
            # Figure source/target
            import json as _json
            try:
                data = _json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[skip] {path.name}: cannot read ({e})")
                continue
            src = norm_lang(data.get("from_language") or lang_code)
            tgt = norm_lang(data.get("to_language") or "en")
            translator = None
            model_used = None
            if trans_engine == "hf":
                candidates = pick_models(src, tgt, args.translation_model)
                for cand in candidates:
                    if cand in trans_cache:
                        translator = trans_cache[cand]
                        model_used = cand
                        break
                if translator is None:
                    model_used, translator = load_translator(candidates)
                    trans_cache[model_used] = translator
            try:
                updated = process_translations(
                    path=path,
                    translator=translator,
                    batch_size=args.batch_size,
                    dry_run=False,
                    only_missing=False,
                    engine=trans_engine,
                    google_api_key=google_key,
                    tgt_lang=tgt or "en",
                )
                print(f"[filled] {path.name}: translations {updated} lines (model={model_used or trans_engine})")
            except Exception as e:
                print(f"[error] {path.name} translation fill failed: {e}")

        print(f"\n=== Filling grammar for {len(new_jsons)} files (engine={gram_engine}) ===")
        gram_cache: Dict[Tuple[str, str], object] = {}
        gram_models = pick_grammar_models(args.grammar_model)
        for path in new_jsons:
            try:
                updated = process_grammar(
                    path=path,
                    generators=gram_cache,
                    model_candidates=gram_models,
                    batch_size=args.batch_size,
                    dry_run=False,
                    engine=gram_engine,
                    local_files_only=False,
                    ollama_model=args.grammar_model or "llama3",
                    ollama_host=get_env("OLLAMA_BASE_URL", "") or ("http://" + get_env("OLLAMA_HOST", "localhost:11434")),
                    only_missing=False,
                )
                print(f"[filled] {path.name}: grammar {updated} lines")
            except Exception as e:
                print(f"[error] {path.name} grammar fill failed: {e}")


if __name__ == "__main__":
    main()
