#!/usr/bin/env python3
"""
Simple retime utility for song JSONs.

Usage examples:

  python scripts/retime_song.py --song-id au_clair_de_la_lune --offset 0.75
  python scripts/retime_song.py --file static/data/frere_jacques.json --factor 1.03
  python scripts/retime_song.py --song-id a_la_claire_fontaine --offset 0.5 --factor 0.98

It updates line times in-place. A backup file with .bak is written once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "static" / "data"


def retime_file(path: Path, offset: float, factor: float) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = data.get("lines") or []
    if not isinstance(lines, list) or not lines:
        raise SystemExit(f"No lines in {path}")

    # Write a one-time backup
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    t_prev = 0.0
    for i, ln in enumerate(lines):
        t = float(ln.get("time", 0.0))
        new_t = offset + factor * t
        if new_t < t_prev:
            new_t = t_prev
        ln["time"] = round(new_t, 3)
        t_prev = ln["time"]

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated {path.name}: offset={offset}, factor={factor}")


def main():
    ap = argparse.ArgumentParser(description="Retime song timestamps")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--song-id", help="Song ID without .json, looked up in static/data")
    g.add_argument("--file", help="Path to a song JSON")
    ap.add_argument("--offset", type=float, default=0.0, help="Seconds to add to each timestamp")
    ap.add_argument("--factor", type=float, default=1.0, help="Multiply all timestamps by this factor")
    args = ap.parse_args()

    if args.file:
        p = Path(args.file)
    else:
        p = DATA_DIR / f"{args.song_id}.json"
    if not p.exists():
        raise SystemExit(f"Not found: {p}")

    retime_file(p, args.offset, args.factor)


if __name__ == "__main__":
    main()

