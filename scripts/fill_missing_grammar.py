#!/usr/bin/env python3
"""Fill missing grammar notes in final song JSON files using local models (HF or Ollama)."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from tqdm import tqdm
from transformers import pipeline, Pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "static" / "data"


def is_grammar_missing(line: Dict[str, Any]) -> bool:
    gram = line.get("grammar")
    if not isinstance(gram, str):
        return True
    gram_clean = gram.strip()
    if not gram_clean:
        return True
    if gram_clean.lower() in {"todo", "tbd", "pending", "grammar"}:
        return True
    return False


def chunked(seq: List[Any], size: int) -> List[List[Any]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def build_prompt(items: List[Tuple[int, Dict[str, Any]]], src_lang: str, tgt_lang: str) -> str:
    # This returns a single-line prompt; we generate per-line for robustness.
    payload = {
        "text": items[0][1].get("text", ""),
        "translation": items[0][1].get("translation", ""),
    }
    return (
        "Write ONE concise grammar/usage note for this lyric line to help learners.\n"
        "- Mention tense/aspect/mood, idiom, agreement, pronouns/particles, or note if it's just an interjection.\n"
        "- Keep it short (1–2 sentences). No markup, no numbering. Respond ONLY with the note text.\n"
        "- Use the target/translation language for the explanation.\n"
        f"Source language: {src_lang or 'unknown'}; Target translation language: {tgt_lang or 'unknown'}.\n"
        f"Line: {json.dumps(payload, ensure_ascii=False)}"
    )


def pick_models(user_model: str | None) -> List[Tuple[str, str]]:
    """Return candidate (model, task) pairs."""
    if user_model:
        name = user_model.lower()
        # Prefer text2text for sequence-to-sequence models (T5/FLAN/UL2/MT0/BART), then fall back to text-generation.
        prefer_t2t = any(k in name for k in ["t5", "flan", "ul2", "mt0", "t0", "bart"])
        if prefer_t2t:
            return [(user_model, "text2text-generation"), (user_model, "text-generation")]
        return [(user_model, "text-generation"), (user_model, "text2text-generation")]
    return [
        ("google/flan-t5-small", "text2text-generation"),
        ("google/flan-t5-base", "text2text-generation"),
        ("Qwen/Qwen2.5-0.5B-Instruct", "text-generation"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "text-generation"),
        ("Orenguteng/LaMini-Flan-T5-783M", "text2text-generation"),
    ]


def load_generator(model_name: str, task: str, local_files_only: bool) -> Pipeline:
    # trust_remote_code helps newer chat models (Qwen, etc.) load without special casing.
    # We don't pass local_files_only into generation kwargs (that caused warnings); rely on env flags instead.
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return pipeline(
        task=task,
        model=model_name,
        device="cpu",  # CPU by default to avoid accelerate/device_map requirements
        trust_remote_code=True,
    )


def clean_note_text(gen_text: str) -> str:
    note = gen_text.strip()
    # Trim any wrapping quotes/backticks
    note = re.sub(r'^["`\s]+', "", note)
    note = re.sub(r'["`\s]+$', "", note)
    return note.strip()


def generate_grammar_single(
    generator: Pipeline,
    task: str,
    line: Dict[str, Any],
    src_lang: str,
    tgt_lang: str,
    context: str,
) -> str:
    prompt = build_prompt([(0, line)], src_lang, tgt_lang)
    if task == "text2text-generation":
        gen_kwargs = dict(max_new_tokens=128)
    else:
        gen_kwargs = dict(max_new_tokens=128, temperature=0.2, top_p=0.9, top_k=50, do_sample=False, return_full_text=True)
    out = generator(prompt, **gen_kwargs)
    first = out[0]
    gen_text = first.get("generated_text") or ""
    if task != "text2text-generation" and isinstance(gen_text, str) and gen_text.startswith(prompt):
        gen_text = gen_text[len(prompt):]
    note = clean_note_text(gen_text)
    if not note:
        raise ValueError(f"{context} returned empty grammar note; raw={gen_text!r}")
    return note.strip()


def generate_grammar_ollama(
    line: Dict[str, Any],
    src_lang: str,
    tgt_lang: str,
    model: str,
    host: str,
    context: str,
) -> str:
    prompt = build_prompt([(0, line)], src_lang, tgt_lang)
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only the grammar/usage note text."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.2},
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    content = ""
    if isinstance(data, dict):
        content = data.get("message", {}).get("content", "") or ""
    elif isinstance(data, list):  # streaming responses aggregated
        content = "".join(chunk.get("message", {}).get("content", "") for chunk in data if isinstance(chunk, dict))
    note = clean_note_text(content)
    if not note:
        raise ValueError(f"{context} (ollama={model}) returned empty grammar note; raw={content!r}")
    return note


def process_file(
    path: Path,
    generators: Dict[Tuple[str, str], Pipeline],
    model_candidates: List[Tuple[str, str]],
    batch_size: int,
    dry_run: bool,
    engine: str,
    local_files_only: bool,
    ollama_model: str,
    ollama_host: str,
    only_missing: bool,
) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[skip] {path.name}: failed to load ({e})")
        return 0

    lines = data.get("lines") or []
    if not isinstance(lines, list) or not lines:
        return 0

    src_lang = data.get("from_language") or "unknown"
    tgt_lang = data.get("to_language") or "en"

    if only_missing:
        target_items = [(idx, line) for idx, line in enumerate(lines) if is_grammar_missing(line)]
    else:
        target_items = [(idx, line) for idx, line in enumerate(lines)]

    if not target_items:
        return 0

    if dry_run:
        # For dry-run mode, just report how many lines need grammar without loading models.
        return len(target_items)

    updated = 0
    for batch in chunked(target_items, batch_size):
        for (idx, line) in batch:
            ctx = f"{path.name} line {idx}"
            note = None
            last_err = None
            if engine == "ollama":
                try:
                    note = generate_grammar_ollama(
                        line,
                        src_lang,
                        tgt_lang,
                        model=ollama_model,
                        host=ollama_host,
                        context=ctx,
                    )
                except Exception as e:
                    last_err = e
                    print(f"[warn] {ctx}: ollama model {ollama_model} failed: {e}")
            else:
                for model_name, task in model_candidates:
                    try:
                        key = (model_name, task)
                        gen = generators.get(key)
                        if gen is None:
                            gen = load_generator(model_name, task, local_files_only=local_files_only)
                            generators[key] = gen
                        note = generate_grammar_single(gen, task, line, src_lang, tgt_lang, context=f"{ctx} [{model_name}/{task}]")
                        break
                    except Exception as e:
                        last_err = e
                        print(f"[warn] {ctx}: model {model_name} ({task}) failed: {e}")
                        continue
            if note is None:
                raise RuntimeError(f"{ctx}: all models failed. last_error={last_err}")
            lines[idx]["grammar"] = note
            updated += 1

    # Final strict check: every line must have grammar
    leftover = [(idx, line) for idx, line in enumerate(lines) if is_grammar_missing(line)]
    if leftover:
        missing_idx = [idx for idx, _ in leftover]
        raise RuntimeError(f"Missing grammar for lines {missing_idx} in {path.name}")

    if not dry_run:
        data["lines"] = lines
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return updated


def main():
    ap = argparse.ArgumentParser(description="Fill missing grammar notes in app JSONs using local HF instruct models.")
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="Directory containing final song JSON files")
    ap.add_argument(
        "--model",
        default=None,
        help="Hugging Face text-generation model name. Defaults to Qwen2.5-0.5B-Instruct then TinyLlama chat.",
    )
    ap.add_argument(
        "--engine",
        choices=["hf", "ollama"],
        default="hf",
        help="Generation backend: Hugging Face pipelines (default) or a local Ollama server.",
    )
    ap.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", "llama3"),
        help="Ollama model name to use when --engine=ollama (default: llama3).",
    )
    ap.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama host URL (default from OLLAMA_HOST env or http://localhost:11434).",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not attempt to download HF models; use cached/local files only (respects HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE).",
    )
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="Only fill lines whose grammar appears missing. By default, all lines are rewritten.",
    )
    ap.add_argument("--batch-size", type=int, default=16, help="Lines per request (unused in per-line mode, kept for compatibility)")
    ap.add_argument("--dry-run", action="store_true", help="Detect and report missing grammar without writing files")
    args = ap.parse_args()

    # Respect offline env flags automatically
    env_offline = os.getenv("HF_HUB_OFFLINE") or os.getenv("TRANSFORMERS_OFFLINE")
    local_files_only = args.local_files_only or (env_offline or "").lower() in {"1", "true", "yes"}

    model_candidates = pick_models(args.model)
    generators: Dict[Tuple[str, str], Pipeline] = {}

    if args.engine == "hf":
        try:
            import torch  # noqa: F401
        except Exception:
            raise SystemExit("PyTorch is required for HF models. Install torch (CPU build is fine) or run with --engine=ollama.")

    target_dir = Path(args.dir)
    files = sorted(target_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {target_dir}")

    total_updates = 0
    for path in tqdm(files, desc="Processing JSONs"):
        try:
            count = process_file(
                path,
                generators,
                model_candidates,
                args.batch_size,
                args.dry_run,
                engine=args.engine,
                local_files_only=local_files_only,
                ollama_model=args.ollama_model,
                ollama_host=args.ollama_host,
                only_missing=args.only_missing,
            )
            if count:
                action = "would update" if args.dry_run else "updated"
                print(f"[{action}] {path.name}: {count} grammar lines")
            total_updates += count
        except Exception as e:
            print(f"[error] {path.name}: {e}")

    print(f"Done. Missing grammar {'detected' if args.dry_run else 'filled'}: {total_updates}")


if __name__ == "__main__":
    main()
