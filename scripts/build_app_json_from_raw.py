#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
import yaml
from util import slugify, estimate_line_timestamps


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw_lyrics"
OUT_DIR = BASE_DIR / "static" / "data"


def slugify_filename(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s_-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_")


def read_header_and_lyrics(path: Path) -> Tuple[Optional[str], Optional[str], str]:
    txt = path.read_text(encoding="utf-8")
    lines = txt.splitlines()
    title = artist = None
    start_idx = 0
    if lines and lines[0].startswith("# "):
        # Header format: "# {title} — {artist}"
        hdr = lines[0][2:].strip()
        # Split on em dash or hyphen
        parts = re.split(r"\s[—-]\s", hdr, maxsplit=1)
        if len(parts) == 2:
            title, artist = parts[0].strip(), parts[1].strip()
        start_idx = 2 if len(lines) > 1 and lines[1].strip() == "" else 1
    body = "\n".join(lines[start_idx:]).strip()
    return title, artist, body


def normalize_lines(raw_lyrics: str) -> List[str]:
    lines = [ln.strip() for ln in raw_lyrics.splitlines()]
    cleaned: List[str] = []
    for ln in lines:
        if not ln:
            if cleaned and cleaned[-1] == "":
                continue
        cleaned.append(ln)
    cleaned = [ln for ln in cleaned if ln]
    return cleaned


def ollama_translate_with_grammar(lines: List[str], src_lang: str, dst_lang: str, base_url: str, model: str) -> List[Dict[str, str]]:
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
    max_chunk = 30
    for i in range(0, len(lines), max_chunk):
        chunk = lines[i : i + max_chunk]
        prompt = mk_prompt(chunk)
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": options,
        }
        # retry a couple times per chunk to reduce partial-song failures
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
                    m = re.search(r"\[.*\]", out, flags=re.DOTALL)
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
        # Normalize length and preserve text order
        if len(parsed) < len(chunk):
            # pad missing items to keep alignment
            for _ in range(len(chunk) - len(parsed)):
                parsed.append({"text": "", "translation": "", "grammar": ""})
        parsed = parsed[: len(chunk)]
        for j, line in enumerate(chunk):
            item = parsed[j] if j < len(parsed) else {"text": "", "translation": "", "grammar": ""}
            item["text"] = line
            item.setdefault("translation", "")
            item.setdefault("grammar", "")
            results.append(item)
        time.sleep(0.2)
    return results


def build_app_json(
    lang_code: str,
    title: str,
    artist: str,
    translated: List[Dict[str, str]],
    track_length: Optional[int] = None,
) -> Dict[str, Any]:
    texts = [item.get("text", "") for item in translated]
    timestamps = estimate_line_timestamps(texts, total_duration=float(track_length) if track_length else None)
    lines_block = []
    for idx, item in enumerate(translated):
        lines_block.append({
            "time": timestamps[idx] if idx < len(timestamps) else float(idx * 3),
            "text": item.get("text", ""),
            "translation": item.get("translation", ""),
            "grammar": item.get("grammar", ""),
        })
    obj = {
        "id": f"{lang_code}_{slugify_filename(title)}",
        "title": title,
        "artist": artist,
        "level": "B2 (Upper Intermediate)",
        "from_language": lang_code,
        "to_language": "en",
        "audio": f"/audio/{lang_code}_{slugify_filename(title)}.mp3",
        "duration": int(track_length) if track_length else max(0, int(timestamps[-1] + 3 if timestamps else len(lines_block) * 3)),
        "licensing_note": "Lyrics fetched via Musixmatch API; translations via local Ollama.",
        "lines": lines_block,
    }
    return obj


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build app JSONs from raw lyrics using Ollama")
    parser.add_argument("--limit", type=int, default=20, help="Number of songs to process")
    parser.add_argument("--fetched-since", help="Only process songs whose metadata fetched_at is on/after this ISO date (YYYY-MM-DD or full timestamp)")
    args = parser.parse_args()

    load_dotenv()
    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    model = os.getenv("OLLAMA_MODEL", "llama3")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build a prioritized list: user's chosen songs first (from data/songs_extra.yaml), then others
    prioritized: List[Path] = []
    extras_path = BASE_DIR / "data" / "songs_extra.yaml"
    if extras_path.exists():
        try:
            extras = yaml.safe_load(extras_path.read_text(encoding="utf-8")) or []
            for item in extras:
                lang = item.get("language")
                title = item.get("title")
                artist = item.get("artist")
                if not (lang and title and artist):
                    continue
                slug = slugify(f"{title}-{artist}")
                candidate = RAW_DIR / lang / f"{slug}.txt"
                if candidate.exists():
                    prioritized.append(candidate)
        except Exception:
            pass

    # Append remaining files not already in prioritized list
    all_files = sorted(RAW_DIR.rglob("*.txt"))
    seen = {p.resolve() for p in prioritized}
    files = prioritized + [p for p in all_files if p.resolve() not in seen]
    def parse_iso(timestamp: str) -> Optional[datetime]:
        if not timestamp:
            return None
        try:
            if timestamp.endswith("Z"):
                timestamp = timestamp[:-1] + "+00:00"
            dt = datetime.fromisoformat(timestamp)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    threshold: Optional[datetime] = None
    if args.fetched_since:
        fs = args.fetched_since.strip()
        if len(fs) == 10:
            fs = fs + "T00:00:00+00:00"
        elif fs.endswith("Z"):
            fs = fs[:-1] + "+00:00"
        try:
            threshold = datetime.fromisoformat(fs)
            if threshold.tzinfo is None:
                threshold = threshold.replace(tzinfo=timezone.utc)
        except Exception:
            raise SystemExit(f"Unable to parse --fetched-since value: {args.fetched_since}")

    processed = 0
    for f in files:
        if processed >= args.limit:
            break
        lang = f.parent.name
        meta_path = f.with_suffix(".meta.json")
        title, artist, body = read_header_and_lyrics(f)
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        title = title or meta.get("title") or f.stem
        artist = artist or meta.get("artist") or ""
        track_length = meta.get("track_length_seconds")
        if threshold:
            fetched_at = meta.get("source", {}).get("fetched_at")
            fetched_dt = parse_iso(fetched_at)
            if not fetched_dt or fetched_dt < threshold:
                continue
        lines = normalize_lines(body)
        if len(lines) < 4:
            continue
        try:
            translated = ollama_translate_with_grammar(lines, lang, "English", base_url, model)
        except Exception as e:
            print(f"Skip {f} due to Ollama error: {e}")
            continue
        obj = build_app_json(lang, title, artist, translated, track_length=track_length)
        out_path = OUT_DIR / f"{lang}_{slugify_filename(title)}.json"
        # Avoid overwriting existing; add suffix if exists
        if out_path.exists():
            k = 2
            while True:
                p2 = OUT_DIR / f"{lang}_{slugify_filename(title)}_{k}.json"
                if not p2.exists():
                    out_path = p2
                    break
                k += 1
        out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {out_path}")
        processed += 1
        time.sleep(0.1)

    print(f"Done. Processed {processed} songs")


if __name__ == "__main__":
    main()
