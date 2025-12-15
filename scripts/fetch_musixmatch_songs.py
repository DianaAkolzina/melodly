#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datetime as dt
import requests
from dotenv import load_dotenv
import sys

from util import slugify as slugify_ascii, estimate_line_timestamps

try:
    # Use langdetect to sanity-check fetched lyric language
    from langdetect import detect, DetectorFactory  # type: ignore

    DetectorFactory.seed = 0
except Exception:  # pragma: no cover - optional dependency
    detect = None  # type: ignore


"""
Fetch new/popular songs with lyrics from Musixmatch by language, translate with
Ollama, and save JSON files matching the app schema in static/data.

Environment:
 - MUSIXMATCH_API_KEY: required
 - OLLAMA_BASE_URL: optional (default http://localhost:11434)
 - OLLAMA_MODEL: optional (default "llama3")

Usage:
  python scripts/fetch_musixmatch_songs.py

Notes:
 - This script respects language filters via Musixmatch track.search with
   f_lyrics_language. For Gaelic, tries 'gd' then 'ga'.
 - It fetches 4 songs per target language (total 20) and skips items with no
   lyrics or restricted lyrics.
 - It asks Ollama to produce translation and brief grammar notes per line.
 - Set your MUSIXMATCH_API_KEY via .env or environment; we do not embed keys.
"""


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "static" / "data"
RAW_DIR = BASE_DIR / "data" / "raw_lyrics"

MUSIXMATCH_BASE = "https://api.musixmatch.com/ws/1.1"

TARGET_LANGS = {
    # Display Name: list of Musixmatch lang codes (try in order)
    "spanish": ["es"],
    "french": ["fr"],
    "gaelic": ["gd", "ga"],
    "latvian": ["lv"],
    "russian": ["ru"],
    "german": ["de"],
    "esperanto": ["eo"],
    "polish": ["pl"],
    "portuguese": ["pt"],
}

PER_LANG_COUNT = 4


@dataclass
class Track:
    track_id: int
    artist_name: str
    track_name: str
    album_name: Optional[str]
    has_lyrics: bool
    commontrack_id: Optional[int]
    track_length: Optional[int] = None
    track_share_url: Optional[str] = None
    track_edit_url: Optional[str] = None


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val or ""


def slugify(text: str) -> str:
    """Create a stable slug; transliterate if needed so Cyrillic/etc. don't collapse to ''. """
    txt = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s_-]", "", txt)
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    if slug:
        return slug
    # Try transliteration if available
    try:
        from unidecode import unidecode  # type: ignore
        slug = unidecode(txt)
        slug = re.sub(r"[^a-z0-9\s_-]", "", slug.lower())
        slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
        if slug:
            return slug
    except Exception:
        pass
    # Fallback: hex hash to keep uniqueness
    return f"track_{abs(hash(text)) & 0xfffffff:x}"


def load_existing_catalog() -> set[tuple[str, str]]:
    """Collect normalized (title, artist) pairs from existing static/data and curated YAMLs."""
    seen: set[tuple[str, str]] = set()
    # Existing JSONs
    for p in DATA_DIR.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            t = (obj.get("title") or "").strip().lower()
            a = (obj.get("artist") or "").strip().lower()
            if t and a:
                seen.add((t, a))
        except Exception:
            pass
    # Curated YAMLs
    for yml in (BASE_DIR / "data").glob("songs*.yaml"):
        try:
            import yaml as _yaml  # lazy
            items = _yaml.safe_load(yml.read_text(encoding="utf-8")) or []
            for it in items:
                t = (it.get("title") or "").strip().lower()
                a = (it.get("artist") or "").strip().lower()
                if t and a:
                    seen.add((t, a))
        except Exception:
            pass
    return seen


def musixmatch_request(endpoint: str, params: Dict[str, Any], api_key: str, timeout: int = 20) -> Dict[str, Any]:
    params = {**params, "apikey": api_key}
    url = f"{MUSIXMATCH_BASE}{endpoint}"
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data


def search_tracks_by_language(lang_code: str, api_key: str, page_size: int = 50, pages: int = 2) -> List[Track]:
    tracks: List[Track] = []
    for page in range(1, pages + 1):
        params = {
            "q_track": "",
            "f_has_lyrics": 1,
            "f_lyrics_language": lang_code,
            # Favor popular and newer items
            "s_track_rating": "desc",
            "s_release_date": "desc",
            "page_size": page_size,
            "page": page,
        }
        payload = musixmatch_request("/track.search", params, api_key)
        msg = payload.get("message", {})
        body = msg.get("body", {})
        lst = body.get("track_list", [])
        for item in lst:
            t = item.get("track", {})
            tracks.append(
                Track(
                    track_id=int(t.get("track_id")),
                    artist_name=t.get("artist_name", "").strip(),
                    track_name=t.get("track_name", "").strip(),
                    album_name=t.get("album_name"),
                    has_lyrics=bool(t.get("has_lyrics", 0)),
                    commontrack_id=t.get("commontrack_id"),
                    track_length=t.get("track_length"),
                    track_share_url=t.get("track_share_url"),
                    track_edit_url=t.get("track_edit_url"),
                )
            )
        # Respect API rate limits a bit
        time.sleep(0.25)
    return tracks


def get_lyrics(track_id: int, api_key: str) -> Optional[str]:
    payload = musixmatch_request("/track.lyrics.get", {"track_id": track_id}, api_key)
    msg = payload.get("message", {})
    body = msg.get("body", {})
    lyrics = body.get("lyrics", {})
    status = msg.get("header", {}).get("status_code")
    if status != 200:
        return None
    text = lyrics.get("lyrics_body") or ""
    # Musixmatch appends disclaimers/truncation markers — remove common footer lines
    text = re.sub(r"\*{2,}.*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    if not text:
        return None
    return text


def normalize_lines(raw_lyrics: str) -> List[str]:
    # Split on newlines, remove empty lines and very short noise-only lines if needed
    lines = [ln.strip() for ln in raw_lyrics.splitlines()]
    # Remove leading/trailing empty lines and dedupe consecutive empties
    cleaned: List[str] = []
    for ln in lines:
        if not ln:
            if cleaned and cleaned[-1] == "":
                continue
        cleaned.append(ln)
    # Drop empty lines
    cleaned = [ln for ln in cleaned if ln]
    return cleaned


def is_language_match(lines: List[str], target_code: str, threshold: float = 0.55) -> bool:
    """Heuristically verify the lines are in the requested language."""
    if not detect or not lines:
        return True  # best effort only if langdetect missing
    codes = []
    for ln in lines[: min(60, len(lines))]:
        try:
            codes.append(detect(ln))
        except Exception:
            continue
    if not codes:
        return True
    target = target_code.split("-")[0].lower()
    hits = sum(1 for c in codes if c == target)
    return (hits / len(codes)) >= threshold


def google_translate_batch(texts: List[str], src: str | None, tgt: str, api_key: str) -> List[str]:
    """Translate a batch of strings using Google Translate API."""
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
    out: List[str] = []
    for t in translations:
        out.append(t.get("translatedText", "") if isinstance(t, dict) else str(t))
    if len(out) != len(texts):
        raise RuntimeError(f"Google Translate returned {len(out)} translations for {len(texts)} inputs")
    return out


def ollama_translate_with_grammar(lines: List[str], src_lang: str, dst_lang: str, base_url: str, model: str) -> List[Dict[str, str]]:
    # To stay robust, chunk lines if too many
    def mk_prompt(chunk: List[str]) -> str:
        joined = "\n".join(chunk)
        return (
            "You are a precise translator and linguist. Translate the following song lines "
            f"from {src_lang} to {dst_lang}. For each original line, produce a JSON array of objects "
            "with keys: text (original line exactly), translation (concise, faithful), grammar (1 short sentence highlighting a relevant grammar or morphology point).\n\n"
            "Return ONLY valid JSON (an array). Do not include any commentary.\n\n"
            "Lines:\n" + joined
    )

    url = f"{base_url.rstrip('/')}/api/generate"
    headers = {"Content-Type": "application/json"}
    options = {"temperature": 0.2, "top_p": 0.9}
    num_gpu_env = os.getenv("OLLAMA_NUM_GPU", "")
    if num_gpu_env:
        try:
            options["num_gpu"] = int(num_gpu_env)
        except ValueError:
            if num_gpu_env.strip().lower() == "cpu":
                options["num_gpu"] = 0
    else:
        options["num_gpu"] = 0

    results: List[Dict[str, str]] = []
    max_chunk = 30  # avoid very long prompts
    for i in range(0, len(lines), max_chunk):
        chunk = lines[i : i + max_chunk]
        prompt = mk_prompt(chunk)
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            # Keep it deterministic-ish for repeatability
            "options": options,
        }
        parsed: List[Dict[str, str]] = []
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=120)
                resp.raise_for_status()
                out = resp.json().get("response", "").strip()
                try:
                    parsed = json.loads(out)
                except json.JSONDecodeError:
                    # Try to extract first JSON array
                    m = re.search(r"\\[.*\\]", out, flags=re.DOTALL)
                    if not m:
                        raise RuntimeError("Ollama did not return valid JSON.")
                    parsed = json.loads(m.group(0))
                if not isinstance(parsed, list):
                    parsed = [parsed]
                break
            except Exception as e:
                last_err = e
                time.sleep(0.5)
        if not parsed and last_err:
            raise last_err
        # Basic validation and alignment
        if len(parsed) < len(chunk):
            for _ in range(len(chunk) - len(parsed)):
                parsed.append({"text": "", "translation": "", "grammar": ""})
        parsed = parsed[: len(chunk)]
        for j, line in enumerate(chunk):
            item = parsed[j] if j < len(parsed) else {"text": "", "translation": "", "grammar": ""}
            # Ensure original text preserved
            item["text"] = line
            item.setdefault("translation", "")
            item.setdefault("grammar", "")
            results.append(item)
        time.sleep(0.2)

    return results


def build_song_json(
    lang_code: str,
    title: str,
    artist: str,
    translated: List[Dict[str, str]],
    to_lang: str = "en",
    track_length: Optional[int] = None,
) -> Dict[str, Any]:
    texts = [item.get("text", "") for item in translated]
    timestamps = estimate_line_timestamps(texts, total_duration=float(track_length) if track_length else None)
    lines_block = []
    for idx, item in enumerate(translated):
        lines_block.append(
            {
                "time": timestamps[idx] if idx < len(timestamps) else float(idx * 3),
                "text": item.get("text", ""),
                "translation": item.get("translation", ""),
                "grammar": item.get("grammar", ""),
            }
        )

    obj = {
        "id": f"{lang_code}_{slugify(title)}",
        "title": title,
        "artist": artist,
        "level": "B2 (Upper Intermediate)",
        "from_language": lang_code,
        "to_language": to_lang,
        "audio": f"/audio/{lang_code}_{slugify(title)}.mp3",
        "duration": int(track_length) if track_length else max(0, int(timestamps[-1] + 3 if timestamps else len(lines_block) * 3)),
        "licensing_note": "Lyrics fetched via Musixmatch API; translations via local Ollama.",
        "lines": lines_block,
    }
    return obj


def ensure_unique_path(base_dir: Path, lang_code: str, title: str) -> Path:
    base = f"{lang_code}_{slugify(title)}"
    p = base_dir / f"{base}.json"
    if not p.exists():
        return p
    # If already exists, add suffix
    k = 2
    while True:
        p = base_dir / f"{base}_{k}.json"
        if not p.exists():
            return p
        k += 1


def fetch_for_language(
    display_lang: str,
    lang_codes: List[str],
    api_key: str,
    ollama_url: str,
    ollama_model: str,
    translation_engine: str,
    google_api_key: Optional[str],
    seen_pairs: set[tuple[str, str]],
    per_count: int,
    skip_annotation: bool,
    verbose: bool = False,
) -> List[Path]:
    saved_paths: List[Path] = []
    for code in lang_codes:
        try:
            candidates = search_tracks_by_language(code, api_key)
        except Exception:
            if verbose:
                print(f"[{display_lang}] search failed for code={code}", file=sys.stderr)
            continue

        for t in candidates:
            if len(saved_paths) >= per_count:
                break
            if not t.has_lyrics:
                if verbose:
                    print(f"[{display_lang}] skip: no lyrics flag for {t.track_name} — {t.artist_name}")
                continue
            # Skip if already present in catalog
            norm_pair = (t.track_name.strip().lower(), t.artist_name.strip().lower())
            if norm_pair in seen_pairs:
                if verbose:
                    print(f"[{display_lang}] skip: already have {t.track_name} — {t.artist_name}")
                continue
            try:
                lyrics = get_lyrics(t.track_id, api_key)
                if not lyrics or len(lyrics.split()) < 5:
                    if verbose:
                        print(f"[{display_lang}] skip: empty/short lyrics for {t.track_name} — {t.artist_name}")
                    continue
                lines = normalize_lines(lyrics)
                if len(lines) < 4:
                    if verbose:
                        print(f"[{display_lang}] skip: too few lines for {t.track_name} — {t.artist_name}")
                    continue
                if not is_language_match(lines, code):
                    if verbose:
                        print(f"[{display_lang}] skip: detected language mismatch for {t.track_name} — {t.artist_name}")
                    continue
                raw_slug = slugify_ascii(f"{t.track_name}-{t.artist_name}")
                if raw_slug == "untitled":
                    raw_slug = f"track-{t.track_id}"
                raw_dir = RAW_DIR / code
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_txt = raw_dir / f"{raw_slug}.txt"
                raw_meta = raw_dir / f"{raw_slug}.meta.json"
                existing_meta = None
                if raw_meta.exists():
                    try:
                        existing_meta = json.loads(raw_meta.read_text(encoding="utf-8"))
                    except Exception:
                        existing_meta = None
                if raw_txt.exists() and raw_meta.exists():
                    if existing_meta and t.track_length and not existing_meta.get("track_length_seconds"):
                        existing_meta["track_length_seconds"] = t.track_length
                        raw_meta.write_text(json.dumps(existing_meta, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(f"Updated existing metadata with track length: {raw_meta}")
                    saved_paths.append(raw_meta)
                    seen_pairs.add(norm_pair)
                    if len(saved_paths) >= per_count:
                        break
                    continue
                header = f"# {t.track_name} — {t.artist_name}\n\n"
                raw_txt.write_text(header + lyrics.strip() + "\n", encoding="utf-8")
                meta = {
                    "language": code,
                    "title": t.track_name,
                    "artist": t.artist_name,
                    "slug": raw_slug,
                    "track_length_seconds": t.track_length,
                    "source": {
                        "provider": "musixmatch",
                        "track_id": t.track_id,
                        "commontrack_id": t.commontrack_id,
                        "track_share_url": t.track_share_url,
                        "track_edit_url": t.track_edit_url,
                        "search_language": code,
                        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                    },
                }
                raw_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                if skip_annotation:
                    print(f"Saved raw lyrics {raw_txt}")
                    saved_paths.append(raw_txt)
                else:
                    if translation_engine == "google":
                        if not google_api_key:
                            raise RuntimeError("GOOGLE_API_KEY is required for translation-engine=google")
                        translations = google_translate_batch(lines, src=code, tgt="en", api_key=google_api_key)
                        translated = [
                            {"text": line, "translation": translations[idx], "grammar": ""}
                            for idx, line in enumerate(lines)
                        ]
                    else:
                        translated = ollama_translate_with_grammar(lines, display_lang, "English", ollama_url, ollama_model)
                    obj = build_song_json(code, t.track_name, t.artist_name, translated, to_lang="en", track_length=t.track_length)
                    out_path = ensure_unique_path(DATA_DIR, code, t.track_name)
                    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"Saved {out_path}")
                    saved_paths.append(out_path)
                seen_pairs.add(norm_pair)
            except Exception as e:
                print(f"Skip track {t.track_name} by {t.artist_name} due to error: {e}")
                time.sleep(0.3)

        if len(saved_paths) >= per_count:
            break
    return saved_paths


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Musixmatch songs and translate with Ollama")
    parser.add_argument("--per-lang", type=int, default=PER_LANG_COUNT, help="Number of songs per language (default: 4)")
    parser.add_argument("--only-lang", action="append", default=None, help="Limit to specific display languages (repeatable). Choices: spanish, french, gaelic, latvian, russian, german, esperanto, polish, portuguese")
    parser.add_argument("--skip-annotation", action="store_true", help="Only fetch raw lyrics and metadata; skip translation/JSON generation")
    parser.add_argument("--verbose", action="store_true", help="Print per-track skip reasons")
    parser.add_argument(
        "--translation-engine",
        choices=["ollama", "google"],
        default="ollama",
        help="Choose translation engine. 'google' uses Google Translate API; 'ollama' uses local model (default).",
    )
    parser.add_argument(
        "--google-api-key",
        default=os.getenv("GOOGLE_API_KEY"),
        help="Google Translate API key (or set GOOGLE_API_KEY env). Required if translation-engine=google.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = get_env("MUSIXMATCH_API_KEY", required=True)
    if args.skip_annotation:
        ollama_url = ""
        ollama_model = ""
    else:
        # Prefer OLLAMA_BASE_URL; fall back to OLLAMA_HOST
        ollama_url = get_env("OLLAMA_BASE_URL", "") or ("http://" + get_env("OLLAMA_HOST", "localhost:11434"))
        ollama_model = get_env("OLLAMA_MODEL", "llama3")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_saved: List[Path] = []
    # Resolve per-language count without mutating the global
    per_count = PER_LANG_COUNT
    if args.per_lang and args.per_lang > 0:
        per_count = args.per_lang

    # Filter languages if requested
    items = list(TARGET_LANGS.items())
    if args.only_lang:
        wanted = set([w.lower() for w in args.only_lang])
        items = [(d, c) for d, c in items if d.lower() in wanted]

    google_key = args.google_api_key
    if args.translation_engine == "google" and not google_key:
        raise SystemExit("GOOGLE_API_KEY is required for translation-engine=google")

    seen_pairs = load_existing_catalog()

    for display, codes in items:
        print(f"=== Fetching {per_count} songs for {display} ({'/'.join(codes)}) ===")
        paths = fetch_for_language(
            display_lang=display,
            lang_codes=codes,
            api_key=api_key,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            translation_engine=args.translation_engine,
            google_api_key=google_key,
            seen_pairs=seen_pairs,
            per_count=per_count,
            skip_annotation=args.skip_annotation,
            verbose=args.verbose,
        )
        total_saved.extend(paths)
        print(f"Saved {len(paths)} for {display}\n")
        # Be polite with rate limits
        time.sleep(1.0)

    print(f"Done. Total new files: {len(total_saved)}")


if __name__ == "__main__":
    main()
