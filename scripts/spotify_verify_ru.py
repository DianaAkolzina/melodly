#!/usr/bin/env python3
"""
Check availability of curated Russian songs on Spotify and optionally delete
local JSONs whose tracks cannot be found.

Focus artists: Zemfira, FACE, Монеточка, Сплин, AIGEL, Нервы.
Default is dry-run: no deletions. Use --remove-missing to delete ru_*.json that
fail lookup.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "static" / "data"


@dataclass
class SongTarget:
    title: str
    artist: str


TARGET_SONGS: list[SongTarget] = [
    SongTarget("Хочешь?", "Земфира"),
    SongTarget("Искала", "Земфира"),
    SongTarget("Аривидерчи", "Земфира"),
    SongTarget("Юморист", "FACE"),
    SongTarget("Бургер", "FACE"),
    SongTarget("Нет Монет", "Монеточка"),
    SongTarget("Каждый раз", "Монеточка"),
    SongTarget("Нимфоманка", "Монеточка"),
    SongTarget("Выхода нет", "Сплин"),
    SongTarget("Мое сердце", "Сплин"),
    SongTarget("Орбит без сахара", "AIGEL"),
    SongTarget("Пыяла", "AIGEL"),
    SongTarget("Батареи", "Нервы"),
    SongTarget("Слёрм", "Нервы"),
]


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"Missing required env var: {name}", file=sys.stderr)
        sys.exit(2)
    return val


def get_spotify_token() -> str:
    cid = _require_env("SPOTIFY_CLIENT_ID")
    secret = _require_env("SPOTIFY_CLIENT_SECRET")
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {auth}"},
        data={"grant_type": "client_credentials"},
        timeout=15,
    )
    if resp.status_code != 200:
        print(f"Failed to get token: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(3)
    return resp.json().get("access_token")


def pick_track(items: list[dict], artist: str) -> Optional[dict]:
    artist_lc = artist.lower()
    best = None
    for t in items:
        names = " ".join([a.get("name", "") for a in t.get("artists", [])]).lower()
        if artist_lc in names or names in artist_lc:
            # Prefer higher popularity
            if best is None or (t.get("popularity", 0) > best.get("popularity", 0)):
                best = t
    if best:
        return best
    return items[0] if items else None


def search_track(session: requests.Session, token: str, title: str, artist: str, market: str) -> Optional[dict]:
    queries = [
        f'track:"{title}" artist:"{artist}"',
        f'"{title}" artist:"{artist}"',
        f"{title} {artist}",
        f'track:"{title}"',
    ]
    headers = {"Authorization": f"Bearer {token}"}
    for q in queries:
        resp = session.get(
            "https://api.spotify.com/v1/search",
            params={"q": q, "type": "track", "limit": 10, "market": market},
            headers=headers,
            timeout=15,
        )
        if resp.status_code != 200:
            continue
        items = resp.json().get("tracks", {}).get("items", [])
        if items:
            return pick_track(items, artist)
    return None


def list_ru_files() -> Iterable[Path]:
    for p in DATA_DIR.glob("ru*.json"):
        if p.is_file():
            yield p


def main():
    ap = argparse.ArgumentParser(description="Verify Russian songs on Spotify and prune missing ones")
    ap.add_argument("--market", default=os.getenv("SPOTIFY_MARKET", "US"), help="Spotify market code, default from SPOTIFY_MARKET or US")
    ap.add_argument("--remove-missing", action="store_true", help="Delete ru*.json files that fail Spotify lookup")
    args = ap.parse_args()

    token = get_spotify_token()
    sess = requests.Session()

    print(f"Using market={args.market}")
    print("Checking curated targets...")
    for s in TARGET_SONGS:
        track = search_track(sess, token, s.title, s.artist, args.market)
        if track:
            artists = ", ".join(a.get("name") for a in track.get("artists", []))
            print(f"[FOUND] {s.artist} — {s.title} | URI {track.get('uri')} | artists: {artists}")
        else:
            print(f"[MISSING] {s.artist} — {s.title}")

    print("\nValidating existing ru*.json files...")
    missing_files: list[Path] = []
    for p in list_ru_files():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            title = str(data.get("title") or "").strip()
            artist = str(data.get("artist") or "").strip()
        except Exception:
            title = p.stem
            artist = ""
        track = search_track(sess, token, title, artist, args.market) if title else None
        if track:
            print(f"[KEEP] {p.name}: {artist} — {title} (uri {track.get('uri')})")
        else:
            print(f"[DROP?] {p.name}: {artist} — {title}")
            missing_files.append(p)

    if args.remove_missing and missing_files:
        for p in missing_files:
            try:
                p.unlink()
                print(f"Deleted {p}")
            except Exception as e:
                print(f"Failed to delete {p}: {e}", file=sys.stderr)
    elif missing_files:
        print(f"\n{len(missing_files)} files flagged missing. Re-run with --remove-missing to delete them.")


if __name__ == "__main__":
    main()
