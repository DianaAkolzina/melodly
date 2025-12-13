#!/usr/bin/env python3
import sys
from pathlib import Path
import yaml


def key(entry):
    return (
        (entry.get("language") or "").strip().lower(),
        (entry.get("title") or "").strip().lower(),
        (entry.get("artist") or "").strip().lower(),
    )


def main():
    base_path = Path("data/songs.yaml")
    extra_path = Path("data/songs_extra.yaml")
    if not base_path.exists() or not extra_path.exists():
        print("Expected data/songs.yaml and data/songs_extra.yaml")
        sys.exit(1)

    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or []
    extra = yaml.safe_load(extra_path.read_text(encoding="utf-8")) or []

    seen = {key(e) for e in base}
    merged = list(base)
    added = 0
    for e in extra:
        k = key(e)
        if k in seen:
            continue
        merged.append(e)
        seen.add(k)
        added += 1

    base_path.write_text(yaml.safe_dump(merged, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Merged: {added} new songs added. Total now: {len(merged)}")


if __name__ == "__main__":
    main()

