#!/usr/bin/env python3
"""
Re-estimate lyric timestamps heuristically.

Example usages:
  python scripts/reestimate_timestamps.py --file static/data/lv_sekundes.json
  python scripts/reestimate_timestamps.py --dir static/data --language lv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List

from util import estimate_line_timestamps


def iter_files(args: argparse.Namespace) -> Iterable[Path]:
    if args.file:
        yield Path(args.file)
        return
    root = Path(args.dir or "static/data")
    if root.is_file():
        yield root
        return
    pattern = "*.json"
    for path in sorted(root.rglob(pattern)):
        if args.language:
            lang_prefix = f"{args.language.lower()}_"
            if not path.name.lower().startswith(lang_prefix):
                continue
        yield path


def update_file(path: Path, dry_run: bool) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines: List[dict] = data.get("lines") or []
    if not isinstance(lines, list) or not lines:
        return False

    texts = [str(ln.get("text", "")) for ln in lines]
    duration = data.get("duration")
    total_duration = float(duration) if isinstance(duration, (int, float)) and duration > 0 else None

    timestamps = estimate_line_timestamps(texts, total_duration=total_duration)
    if not timestamps:
        return False

    changed = False
    for idx, line in enumerate(lines):
        new_time = round(timestamps[idx], 3) if idx < len(timestamps) else round(timestamps[-1], 3)
        old_time = float(line.get("time", 0.0))
        if abs(new_time - old_time) > 1e-3:
            line["time"] = new_time
            changed = True

    if not changed:
        return False

    last_time = timestamps[-1]
    suggested_duration = math.ceil(last_time + 2.0)
    if not total_duration or suggested_duration > total_duration:
        data["duration"] = suggested_duration

    if dry_run:
        return True

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-estimate lyric timestamps heuristically")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Process a single JSON file")
    group.add_argument("--dir", help="Process all JSON files under this directory (default: static/data)")
    ap.add_argument("--language", help="Only process JSON files whose filename starts with this language code (e.g., es)")
    ap.add_argument("--dry-run", action="store_true", help="Report files that would change without writing them")
    args = ap.parse_args()

    total = 0
    changed = 0
    for path in iter_files(args):
        total += 1
        if update_file(path, dry_run=args.dry_run):
            changed += 1
            print(f"Updated timestamps: {path}")

    if total == 0:
        print("No matching files found.")
    elif args.dry_run:
        print(f"Dry run complete. {changed} / {total} files would change.")
    else:
        print(f"Done. Updated {changed} / {total} files.")


if __name__ == "__main__":
    main()
