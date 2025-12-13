#!/usr/bin/env python3
import os
import json
import time
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from dotenv import load_dotenv
import yaml

from util import slugify, ensure_dir


MUSIXMATCH_API = "https://api.musixmatch.com/ws/1.1"


def mm_get(endpoint: str, params: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    params = {**params, "apikey": api_key}
    r = requests.get(f"{MUSIXMATCH_API}/{endpoint}", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("body", {})


def mm_search_track(title: str, artist: str, api_key: str) -> Optional[Dict[str, Any]]:
    body = mm_get(
        "track.search",
        {
            "q_track": title,
            "q_artist": artist,
            "f_has_lyrics": 1,
            "s_track_rating": "desc",
            "page_size": 1,
        },
        api_key,
    )
    lst = body.get("track_list") or []
    if not lst:
        return None
    return lst[0].get("track")


def mm_get_lyrics(track_id: int, api_key: str) -> Optional[str]:
    body = mm_get("track.lyrics.get", {"track_id": track_id}, api_key)
    lyr = body.get("lyrics", {}).get("lyrics_body")
    if not lyr:
        return None
    # Strip Musixmatch disclaimer and trailing boilerplate if present
    lines = []
    for line in lyr.splitlines():
        if line.strip().startswith("******* This Lyrics is NOT for Commercial use"):
            break
        lines.append(line.rstrip())
    return "\n".join(lines).rstrip()


def main():
    # load .env if present
    load_dotenv()
    api_key = os.getenv("MUSIXMATCH_API_KEY")
    if not api_key:
        raise SystemExit("Please set MUSIXMATCH_API_KEY in your environment (or .env)")

    songs_path = Path("data/songs.yaml")
    if not songs_path.exists():
        raise SystemExit("data/songs.yaml not found")

    songs = yaml.safe_load(songs_path.read_text(encoding="utf-8"))
    out_root = Path("data/raw_lyrics")

    for idx, s in enumerate(songs, 1):
        lang = s.get("language")
        title = s.get("title")
        artist = s.get("artist")
        if not (lang and title and artist):
            print(f"[{idx:02}] Skipping incomplete entry: {s}")
            continue

        slug = slugify(f"{title}-{artist}")
        out_txt = out_root / lang / f"{slug}.txt"
        out_meta = out_root / lang / f"{slug}.meta.json"
        ensure_dir(out_txt)

        # Skip if already fetched
        if out_txt.exists() and out_meta.exists():
            print(f"[{idx:02}] Exists: {title} — {artist}")
            continue

        print(f"[{idx:02}] Searching: {title} — {artist}")
        try:
            track = mm_search_track(title, artist, api_key)
        except requests.HTTPError as e:
            print(f"  HTTP error on search: {e}")
            time.sleep(1.2)
            continue

        if not track:
            print("  No track found with lyrics.")
            time.sleep(0.5)
            continue

        track_id = track.get("track_id")
        print(f"  Found track_id={track_id}, fetching lyrics…")
        try:
            lyrics = mm_get_lyrics(track_id, api_key)
        except requests.HTTPError as e:
            print(f"  HTTP error on lyrics: {e}")
            time.sleep(1.2)
            continue

        if not lyrics:
            print("  No lyrics returned.")
            time.sleep(0.5)
            continue

        out_txt.write_text(
            f"# {title} — {artist}\n\n" + lyrics + "\n",
            encoding="utf-8",
        )

        meta = {
            "language": lang,
            "title": title,
            "artist": artist,
            "slug": slug,
            "source": {
                "provider": "musixmatch",
                "track_id": track_id,
                "track_share_url": track.get("track_share_url"),
                "track_edit_url": track.get("track_edit_url"),
                "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
            },
        }
        out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved: {out_txt}")

        # Gentle rate limit
        time.sleep(0.6)


if __name__ == "__main__":
    main()
