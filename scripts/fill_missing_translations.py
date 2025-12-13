#!/usr/bin/env python3
"""Fill missing line translations in final app JSON files using free Hugging Face translation models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable

import requests
from tqdm import tqdm
from transformers import pipeline, Pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "static" / "data"


def is_translation_missing(line: Dict[str, Any]) -> bool:
    """Heuristically determine if a line needs translation."""
    tr = line.get("translation")
    txt = (line.get("text") or "").strip()
    if not isinstance(tr, str):
        return True
    tr_clean = tr.strip()
    if not tr_clean:
        return True
    if tr_clean.lower() in {"todo", "tbd", "pending", "translate"}:
        return True
    # Treat copies of the source text as missing (case/punctuation agnostic)
    def _norm(s: str) -> str:
        import re
        return re.sub(r"[\\W_]+", "", s.lower())
    if txt and (_norm(tr_clean) == _norm(txt)):
        return True
    return False


def chunked(seq: List[Any], size: int) -> List[List[Any]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def norm_lang(code: str | None) -> str:
    return (code or "").split("-")[0].strip().lower()


def pick_models(src: str, tgt: str, user_model: str | None) -> List[str]:
    """Return ordered candidate models to try, favoring language-specific Marian MT where available."""
    if user_model:
        return [user_model]
    per_lang: dict[str, list[str]] = {
        # High-quality, language-specific Marian models
        "fr": ["Helsinki-NLP/opus-mt-fr-en"],
        "es": ["Helsinki-NLP/opus-mt-es-en"],
        "pt": ["Helsinki-NLP/opus-mt-pt-en"],
        "it": ["Helsinki-NLP/opus-mt-it-en"],
        "de": ["Helsinki-NLP/opus-mt-de-en"],
        "lv": ["Helsinki-NLP/opus-mt-lv-en"],
        "ga": ["Helsinki-NLP/opus-mt-ga-en"],  # Irish
        "gd": ["Helsinki-NLP/opus-mt-gd-en"],  # Scottish Gaelic
        "ru": ["Helsinki-NLP/opus-mt-ru-en"],
    }
    src_norm = norm_lang(src)
    if tgt == "en" and src_norm in per_lang:
        return per_lang[src_norm] + ["Helsinki-NLP/opus-mt-mul-en", "Helsinki-NLP/opus-mt-mul-mul"]
    candidates: List[str] = []
    if src_norm and tgt:
        candidates.append(f"Helsinki-NLP/opus-mt-{src_norm}-{tgt}")
    if tgt == "en":
        candidates.append("Helsinki-NLP/opus-mt-mul-en")
    candidates.append("Helsinki-NLP/opus-mt-mul-mul")
    return candidates


def google_detect_language(text: str, api_key: str) -> str:
    """Detect language using Google Translate API; returns ISO language code or ''."""
    url = f"https://translation.googleapis.com/language/translate/v2/detect?key={api_key}"
    try:
        resp = requests.post(url, json={"q": text}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        detections = data.get("data", {}).get("detections", [])
        if detections and detections[0]:
            # detections[0] is a list of candidates
            cand = detections[0][0]
            lang = cand.get("language") or ""
            return norm_lang(lang)
    except Exception as e:
        print(f"[warn] detect failed: {e}")
    return ""


def google_translate_batch(texts: List[str], src: str | None, tgt: str, api_key: str) -> List[str]:
    url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
    payload: Dict[str, Any] = {
        "q": texts,
        "target": tgt or "en",
        "format": "text",
    }
    if src:
        payload["source"] = src
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    translations = data.get("data", {}).get("translations", [])
    out = []
    for t in translations:
        out.append(t.get("translatedText", "") if isinstance(t, dict) else str(t))
    if len(out) != len(texts):
        raise RuntimeError(f"Google Translate returned {len(out)} translations for {len(texts)} inputs")
    return out


def load_translator(candidates: List[str]) -> Tuple[str, Pipeline]:
    last_err: Exception | None = None
    for name in candidates:
        try:
            return name, pipeline("translation", model=name)
        except Exception as e:
            last_err = e
            continue
    raise SystemExit(f"Failed to load any translation model from {candidates}: {last_err}")


def translate_batch(
    translator: Pipeline,
    items: List[Tuple[int, str]],
) -> List[str]:
    """Translate a batch of lyric lines, returning translations in order."""
    texts = [text for _, text in items]
    outputs = translator(texts, clean_up_tokenization_spaces=True)
    return [o["translation_text"] if isinstance(o, dict) else str(o) for o in outputs]


def process_file(
    path: Path,
    translator: Pipeline | None,
    batch_size: int,
    dry_run: bool,
    only_missing: bool,
    engine: str,
    google_api_key: str | None,
    tgt_lang: str,
) -> int:
    """Process one JSON file; return number of lines updated."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[skip] {path.name}: failed to load ({e})")
        return 0

    lines = data.get("lines") or []
    if not isinstance(lines, list) or not lines:
        return 0

    if engine == "google":
        if google_api_key is None:
            raise RuntimeError("GOOGLE_API_KEY is required for --engine=google")
        # Detect language on the whole text blob to catch mislabelled files
        blob = "\n".join([(ln.get("text") or "").strip() for ln in lines if (ln.get("text") or "").strip()])
        detected = google_detect_language(blob[:4000], google_api_key)
        src_norm = norm_lang(data.get("from_language"))
        if detected and detected != src_norm:
            print(f"[fix] {path.name}: detected language {detected} != from_language {src_norm or '(none)'}; updating from_language to {detected}")
            data["from_language"] = detected
            src_norm = detected
        if only_missing:
            targets = [(idx, line.get("text") or "") for idx, line in enumerate(lines) if is_translation_missing(line)]
        else:
            targets = [(idx, line.get("text") or "") for idx, line in enumerate(lines) if (line.get("text") or "").strip()]
        if not targets:
            return 0
        updated = 0
        for batch in chunked(targets, batch_size):
            texts = [text for _, text in batch]
            translations = google_translate_batch(texts, src_norm or None, tgt_lang, google_api_key)
            for (idx, _src), tr in zip(batch, translations):
                lines[idx]["translation"] = tr
                updated += 1
        if dry_run:
            return updated
        data["lines"] = lines
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return updated

    # HF engine path
    if only_missing:
        targets = [(idx, line.get("text") or "") for idx, line in enumerate(lines) if is_translation_missing(line)]
    else:
        targets = [(idx, line.get("text") or "") for idx, line in enumerate(lines) if (line.get("text") or "").strip()]

    if not targets:
        return 0

    updated = 0
    for batch in chunked(targets, batch_size):
        translations = translate_batch(translator, batch)
        if len(translations) != len(batch):
            raise RuntimeError(f"Model returned {len(translations)} translations for {len(batch)} items")
        for (idx, _src), tr in zip(batch, translations):
            lines[idx]["translation"] = tr
            updated += 1

    if not dry_run:
        data["lines"] = lines
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return updated


def main():
    ap = argparse.ArgumentParser(description="Translate lyric lines in app JSONs using Google Translate or MarianMT.")
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="Directory containing final song JSON files")
    ap.add_argument(
        "--model",
        default=None,
        help="Hugging Face translation model name. Default auto-picks Helsinki-NLP/opus-mt-<src>-<tgt> or mul-en.",
    )
    ap.add_argument(
        "--engine",
        choices=["google", "hf"],
        default="google",
        help="Translation engine: Google Translate API (default) or Hugging Face MarianMT.",
    )
    ap.add_argument(
        "--google-api-key",
        default=os.getenv("GOOGLE_API_KEY"),
        help="Google Translate API key (or set GOOGLE_API_KEY env).",
    )
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="Only translate lines that look missing. By default, all lines are retranslated.",
    )
    ap.add_argument("--batch-size", type=int, default=12, help="Lines per translation request")
    ap.add_argument("--dry-run", action="store_true", help="Detect and report missing translations without writing files")
    args = ap.parse_args()

    target_dir = Path(args.dir)
    files = sorted(target_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {target_dir}")

    total_updates = 0
    cached_translators: dict[str, Pipeline] = {}
    for path in tqdm(files, desc="Processing JSONs"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            src = norm_lang(data.get("from_language") or "")
            tgt = norm_lang(data.get("to_language") or "en")
            model_used = None
            translator: Pipeline | None = None

            if args.engine == "hf":
                model_candidates = pick_models(src, tgt, args.model)
                for cand in model_candidates:
                    if cand in cached_translators:
                        translator = cached_translators[cand]
                        model_used = cand
                        break
                if translator is None:
                    model_used, translator = load_translator(model_candidates)
                    cached_translators[model_used] = translator
                if translator is None:
                    raise RuntimeError("translator init failed")

            count = process_file(
                path,
                translator,
                args.batch_size,
                args.dry_run,
                only_missing=args.only_missing,
                engine=args.engine,
                google_api_key=args.google_api_key,
                tgt_lang=tgt or "en",
            )
            if count:
                action = "would update" if args.dry_run else "updated"
                model_label = model_used or "google"
                print(f"[{action}] {path.name}: {count} lines (model={model_label})")
            total_updates += count
        except Exception as e:
            print(f"[error] {path.name}: {e}")

    print(f"Done. Translations {'checked' if args.dry_run else 'filled'}: {total_updates}")


if __name__ == "__main__":
    main()
