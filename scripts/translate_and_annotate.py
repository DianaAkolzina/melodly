#!/usr/bin/env python3
import os
import json
from pathlib import Path
from typing import Dict, Any

import yaml
import requests
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


def load_songs_index() -> Dict[str, Dict[str, Any]]:
    songs = yaml.safe_load(Path("data/songs.yaml").read_text(encoding="utf-8"))
    index = {}
    for s in songs:
        key = (s.get("language"), s.get("title"), s.get("artist"))
        index["|".join(key)] = s
    return index


def build_prompt(lang: str, lyrics: str) -> str:
    return (
        "You are a precise translator and linguist.\n"
        f"Source language code: {lang}. Translate the following song lyrics into natural, idiomatic English.\n"
        "Then extract a glossary of 15-30 useful words/phrases with part-of-speech and brief gloss.\n"
        "Provide targeted grammar notes (3-6) with short explanations and 1-2 cited examples from the lyrics.\n"
        "Also produce approximate timing for each non-empty line in the lyrics, assuming a typical 3:00–4:00 duration if unknown; distribute times evenly by line order.\n\n"
        "Return strict JSON with keys: translation.full_text, glossary[], grammar[], timestamps[].\n"
        "- glossary item: {source, pos, gloss, notes?}\n"
        "- grammar item: {topic, explanation, examples: [{source, translation}]}\n"
        "- timestamp item: {index, text, approx_start: 'mm:ss'} (no extra keys)\n\n"
        "Lyrics (preserve original line order):\n" + lyrics
    )


def call_openai(model: str, prompt: str) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package not available; install openai>=1.0.0")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment (or .env)")
    client = OpenAI(api_key=api_key)
    # Use chat.completions for broad compatibility
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You return only valid JSON and nothing else."},
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content or "{}"
    return json.loads(text)


def call_ollama(model: str, prompt: str) -> Dict[str, Any]:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "format": "json",
        "messages": [
            {"role": "system", "content": "You return only valid JSON and nothing else."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.2},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # streaming vs non-streaming responses: support both
    if isinstance(data, dict) and "message" in data and data.get("done"):
        content = data.get("message", {}).get("content", "{}")
        return json.loads(content or "{}")
    # If streaming, the endpoint might return multiple JSON lines; try to join
    if isinstance(data, list):
        content = "".join([chunk.get("message", {}).get("content", "") for chunk in data])
        return json.loads(content or "{}")
    # Fallback
    content = data.get("message", {}).get("content", "{}") if isinstance(data, dict) else "{}"
    return json.loads(content or "{}")


def detect_mixed_language(engine: str, model: str, lang: str, lyrics: str, threshold: float = 0.25) -> bool:
    """Return True if a significant portion of lines appear not to be in the target language.

    Uses the selected LLM to estimate the fraction of non-target-language lines.
    """
    instr = (
        "Estimate the fraction of lines in the following lyrics that are not in the target language.\n"
        f"Target language code: {lang}.\n"
        "Return JSON: {fraction_not_target: number between 0 and 1}.\n\n"
        "Lyrics:\n" + lyrics
    )
    if engine == "openai":
        result = call_openai(model, instr)
    else:
        result = call_ollama(model, instr)
    try:
        frac = float(result.get("fraction_not_target", 0))
    except Exception:
        frac = 0.0
    return frac >= threshold


def main():
    load_dotenv()
    engine = os.getenv("ENGINE", "openai").lower()
    model = (
        os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if engine == "openai"
        else os.getenv("OLLAMA_MODEL", "llama3")
    )
    raw_root = Path("data/raw_lyrics")
    out_root = Path("data/processed")
    songs_index = load_songs_index()

    for txt in raw_root.rglob("*.txt"):
        lang = txt.parents[0].name
        meta_path = txt.with_suffix(".meta.json")
        if not meta_path.exists():
            print(f"Skipping (no meta): {txt}")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        title = meta.get("title")
        artist = meta.get("artist")
        slug = meta.get("slug")
        lyrics = txt.read_text(encoding="utf-8")
        # Drop header line if present
        if lyrics.startswith("# "):
            lyrics = "\n".join(lyrics.splitlines()[2:])

        # Optional mixed-language filter
        if os.getenv("SKIP_MIXED_LANG", "0") in ("1", "true", "True"):
            try:
                if detect_mixed_language(engine, model, lang, lyrics):
                    print("  Skipped due to mixed-language lyrics above threshold.")
                    continue
            except Exception as e:
                print(f"  Language mix detection failed, proceeding: {e}")

        prompt = build_prompt(lang, lyrics)
        print(f"Translating + annotating: {title} — {artist}")
        try:
            if engine == "openai":
                result = call_openai(model, prompt)
            elif engine == "ollama":
                result = call_ollama(model, prompt)
            else:
                raise RuntimeError(f"Unknown ENGINE: {engine}")
        except Exception as e:
            print(f"  Translation/annotation failed: {e}")
            continue

        package = {
            "language": lang,
            "title": title,
            "artist": artist,
            "slug": slug,
            "lyrics_path": str(txt),
            "source": meta.get("source"),
            "translation": {
                "language": "en",
                "full_text": result.get("translation", {}).get("full_text", ""),
            },
            "glossary": result.get("glossary", []),
            "grammar": result.get("grammar", []),
            "timestamps": result.get("timestamps", []),
        }

        out_path = out_root / lang / f"{slug}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
