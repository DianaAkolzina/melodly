#!/usr/bin/env python3
from __future__ import annotations

"""
Mic → Whisper (free, local) → Align known lyrics → Update song JSON.

This records from your microphone while a track plays on your speakers (or any source),
transcribes with Faster-Whisper, aligns to the lyrics in static/data/<song_id>.json,
and writes updated per-line start times back into the same JSON.

Dependencies (install locally):
  pip install faster-whisper sounddevice soundfile rapidfuzz

Usage examples:
  # Record 90 seconds from mic and align
  python scripts/mic_whisper_align.py --song-id frere_jacques --duration 90 --model small

  # Use an existing audio file instead of the mic
  python scripts/mic_whisper_align.py --song-id frere_jacques --input /path/to/capture.wav --model small

Notes:
  - Faster-Whisper runs fully local (free). CPU works; GPU is faster if available.
  - For best results, play the song clearly on speakers in a quiet room.
  - The alignment is tolerant but works best when lyrics match the sung version.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import List, Tuple

# Lazy imports with helpful messages
try:
    import sounddevice as sd  # type: ignore
    import soundfile as sf    # type: ignore
except Exception:
    sd = None  # type: ignore
    sf = None  # type: ignore

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore

try:
    from rapidfuzz.fuzz import ratio  # type: ignore
except Exception:
    ratio = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "static" / "data"
LOG_DIR = ROOT / "data"
LOG_PATH = LOG_DIR / "timestamps.log"
CAPTURE_DIR = LOG_DIR / "captures"
JOB_PATH = LOG_DIR / "timestamps_job.json"


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    # strip accents
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # keep letters/digits/spaces only
    cleaned = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
    return ' '.join(''.join(cleaned).split())


def _log_line(msg: str) -> None:
    ts = int(time.time())
    line = f"{ts} {msg}\n"
    try:
        LOG_DIR.mkdir(exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    # also echo to stdout for interactive runs
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _read_job() -> dict:
    try:
        if JOB_PATH.exists():
            return json.loads(JOB_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_job(update: dict) -> None:
    try:
        cur = _read_job()
        cur.update(update)
        tmp = JOB_PATH.with_suffix(JOB_PATH.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, JOB_PATH)
    except Exception:
        pass


def _pick_default_input_device() -> int | None:
    """Return a usable input device index or None if not found."""
    try:
        devs = sd.query_devices()
    except Exception:
        return None
    # Prefer default input if valid
    try:
        default_in, _ = sd.default.device
        if isinstance(default_in, int) and default_in >= 0:
            info = sd.query_devices(default_in, 'input')
            if info and info.get('max_input_channels', 0) > 0:
                return default_in
    except Exception:
        pass
    # Fallback: first device with input channels
    for i, d in enumerate(devs):
        if d.get('max_input_channels', 0) > 0:
            return i
    return None


def record_to_wav(out_path: Path, duration: float, samplerate: int | None = None, channels: int = 1, device: int | str | None = None, hostapi: int | None = None) -> None:
    if sd is None or sf is None:
        print("Missing sounddevice/soundfile. Install: pip install sounddevice soundfile", file=sys.stderr)
        sys.exit(1)
    # Resolve device
    if hostapi is not None:
        try:
            sd.default.hostapi = hostapi
        except Exception:
            pass
    if device is None:
        picked = _pick_default_input_device()
        if picked is None:
            print("No input device found. Use --list-devices and --device to select one.", file=sys.stderr)
            sys.exit(2)
        device = picked
    # Derive samplerate from device if not provided
    if samplerate is None:
        try:
            info = sd.query_devices(device, 'input')
            samplerate = int(info.get('default_samplerate') or 16000)
        except Exception:
            samplerate = 16000

    _log_line(f"RECORD start duration={duration:.1f}s device={device} samplerate={samplerate} channels={channels}")
    _write_job({"phase": "recording"})
    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float32', device=device)
    except Exception as e:
        _log_line(f"ERROR failed_to_start_recording err={e}")
        print("Failed to start recording:", e, file=sys.stderr)
        print("Tip: run 'python -m sounddevice' or use '--list-devices' and pick a device index via '--device'.", file=sys.stderr)
        sys.exit(3)
    sd.wait()
    sf.write(str(out_path), audio, samplerate)
    try:
        size = out_path.stat().st_size
    except Exception:
        size = -1
    _log_line(f"RECORD saved path={out_path} size={size}")
    _write_job({"phase": "transcribing", "capture_path": str(out_path)})


def transcribe_words(wav_path: Path, model_size: str = "small", whisper_device: str = "auto", compute_type: str | None = None) -> List[Tuple[str, float, float]]:
    if WhisperModel is None:
        print("Missing faster-whisper. Install: pip install faster-whisper", file=sys.stderr)
        sys.exit(1)
    _log_line(f"TRANSCRIBE start model={model_size} device={whisper_device} compute={compute_type or 'default'} file={wav_path}")
    _write_job({"phase": "loading_model"})
    # Try requested device, fallback to CPU if it fails
    try:
        t0 = time.time()
        model = WhisperModel(model_size, device=whisper_device, compute_type=compute_type or "auto")
        _log_line(f"MODEL ready in {time.time()-t0:.1f}s device={whisper_device} compute={compute_type or 'auto'}")
    except Exception as e:
        _log_line(f"TRANSCRIBE model_init_error err={e}; falling_back_to_cpu")
        t0 = time.time()
        model = WhisperModel(model_size, device="cpu")
        _log_line(f"MODEL ready (CPU fallback) in {time.time()-t0:.1f}s")
    _write_job({"phase": "decoding"})
    t_decode0 = time.time()

    # Inspect audio level to warn if extremely quiet
    total = 0.0
    rms_db = None
    try:
        if sf is not None:
            meta = sf.info(str(wav_path))
            total = float(getattr(meta, 'duration', 0.0) or 0.0)
            # Rough RMS sample to detect very quiet captures
            with sf.SoundFile(str(wav_path)) as f:
                n = min(10 * int(f.samplerate or 16000), len(f))
                if n > 0:
                    import numpy as _np
                    data = f.read(n, dtype='float32', always_2d=False)
                    if data is not None and len(data):
                        arr = _np.asarray(data)
                        if arr.ndim > 1:
                            arr = arr.mean(axis=1)
                        rms = float(_np.sqrt(_np.mean(arr**2)) + 1e-12)
                        rms_db = 20.0 * _np.log10(rms)
    except Exception:
        pass
    if rms_db is not None and rms_db < -30.0:
        _log_line(f"AUDIO warning low_level rms_db={rms_db:.1f}dBFS — raise speaker volume for better results")

    def run_decode(vad: bool, note: str) -> tuple[list[tuple[str,float,float]], float, int]:
        segs, _info = model.transcribe(
            str(wav_path),
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": 300} if vad else None,
            word_timestamps=True,
            temperature=0.2,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.8,
            no_speech_threshold=0.2,
            beam_size=5,
        )
        words_acc: List[Tuple[str, float, float]] = []
        last_log_t = -10.0
        seg_idx = 0
        start_t = time.time()
        for seg in segs:
            seg_idx += 1
            for w in seg.words or []:
                if w.word and w.start is not None and w.end is not None:
                    words_acc.append((w.word, float(w.start), float(w.end)))
            cur_t = float(getattr(seg, 'end', 0.0) or 0.0)
            if total <= 0.0:
                # infer total if unknown
                nonlocal_total = cur_t
            if cur_t - last_log_t >= 10.0 or seg_idx % 5 == 0:
                pct = (100.0 * cur_t / total) if total > 0 else 0.0
                _log_line(f"TRANSCRIBE progress ({note}) t={cur_t:.1f}s/{total:.1f}s ({pct:.0f}%) segments={seg_idx} words={len(words_acc)}")
                last_log_t = cur_t
        elapsed = time.time() - start_t
        return words_acc, (words_acc[-1][2] if words_acc else 0.0), seg_idx

    # First pass with VAD to skip long silences
    words: List[Tuple[str, float, float]]
    words, last_time, nseg = run_decode(vad=True, note="vad")
    coverage = (last_time / total) if total > 0 else 0.0
    if (coverage < 0.6) or (len(words) < 30 and total > 150):
        _log_line(f"TRANSCRIBE fallback running full decode (coverage={coverage:.2f}, words={len(words)})")
        words, last_time, nseg = run_decode(vad=False, note="full")
    if not words:
        _log_line("TRANSCRIBE no_words_detected")
        print("No words detected. Try increasing duration or playing louder.")
    else:
        _log_line(f"TRANSCRIBE done words={len(words)} in {time.time()-t_decode0:.1f}s (coverage={(last_time/total if total>0 else 0.0):.2f})")
    return words


def align_lines(lyrics_lines: List[str], words: List[Tuple[str, float, float]]) -> List[float]:
    """Return per-line start time (seconds) using fuzzy matching with continuity constraints.
    Avoids locking onto late repeated choruses by limiting jumps and penalizing distance.
    """
    if ratio is None:
        print("Missing rapidfuzz. Install: pip install rapidfuzz", file=sys.stderr)
        sys.exit(1)

    # Normalize recognized words and keep mapping to original indices and times
    norm_words = [normalize_text(w[0]) for w in words]
    filtered_to_orig = [i for i, w in enumerate(norm_words) if w]
    joined_words = [norm_words[i] for i in filtered_to_orig]
    time_at_filt = [float(words[orig_i][1]) for orig_i in filtered_to_orig]

    line_times: List[float] = []
    prev_j = 0
    N = len(joined_words)
    total_audio = time_at_filt[-1] if time_at_filt else 0.0
    avg_line_gap = max(0.8, total_audio / max(1, len(lyrics_lines)))
    max_jump_sec = 12.0  # guard against huge leaps when match is weak

    # Tunables
    primary_band = 80        # search range in words ahead of prev_j
    expand_band = 400        # maximum expansion if poor match
    jump_penalty = 0.20      # penalty per primary_band distance (scaled to fuzz 0..100)
    good_threshold = 88      # accept early if score is strong
    min_threshold = 72       # expand if below this

    _write_job({"phase": "aligning"})
    for idx, line in enumerate(lyrics_lines):
        target = normalize_text(line)
        if not target:
            # Empty line; inherit previous
            line_times.append(line_times[-1] if line_times else 0.0)
            _log_line(f"ALIGN line={idx+1}/{len(lyrics_lines)} time={line_times[-1]:.3f} text=(blank)")
            continue

        # Short-line disambiguation: include the next line when current target is very short
        tokens = target.split()
        approx_len = max(2, min(12, len(tokens)))
        if len(tokens) <= 2 and idx + 1 < len(lyrics_lines):
            nxt = normalize_text(lyrics_lines[idx + 1] or "")
            target_for_match = f"{target} {nxt}" if nxt else target
            approx_len = min(14, max(approx_len, len(target_for_match.split()) // 2))
        else:
            target_for_match = target

        best = {"score": -1.0, "j": None, "raw": -1.0}
        def consider_window(lo: int, hi: int, penalty_scale: float):
            nonlocal best
            hi = min(N, hi)
            for j in range(lo, hi):
                k = min(N, j + approx_len)
                window_text = ' '.join(joined_words[j:k])
                sc_raw = float(ratio(target_for_match, window_text))
                dist_units = max(0, j - prev_j) / max(1.0, float(primary_band))
                sc = sc_raw - penalty_scale * jump_penalty * 100.0 * dist_units
                if sc > best["score"]:
                    best = {"score": sc, "j": j, "raw": sc_raw}
                if sc_raw >= good_threshold:
                    break

        # Primary narrow search
        consider_window(prev_j, prev_j + primary_band, penalty_scale=1.0)
        # Expand if needed
        if best["score"] < min_threshold:
            consider_window(prev_j, prev_j + expand_band, penalty_scale=1.5)

        # Bias the first line toward the earliest reasonable match to avoid locking onto a late chorus
        if idx == 0 and best["j"] is not None:
            early_cap = min(N, primary_band * 2)
            earliest_pref = {"score": -1.0, "j": None, "raw": -1.0}
            def consider_early(lo: int, hi: int):
                nonlocal earliest_pref
                hi = min(N, hi)
                for j2 in range(lo, hi):
                    k2 = min(N, j2 + approx_len)
                    window_text = ' '.join(joined_words[j2:k2])
                    sc_raw = float(ratio(target_for_match, window_text))
                    if sc_raw > earliest_pref["score"]:
                        earliest_pref = {"score": sc_raw, "j": j2, "raw": sc_raw}
            consider_early(0, early_cap)
            if earliest_pref["j"] is not None:
                far = best["j"] > primary_band * 2
                margin = 5.0
                if far or earliest_pref["score"] >= best["score"] - margin:
                    _log_line(f"ALIGN first_line_early_bias score={earliest_pref['score']:.1f} raw={earliest_pref['raw']:.1f} j={earliest_pref['j']}")
                    best = earliest_pref
                    prev_j = 0

        if best["j"] is None or (best["score"] < min_threshold and best["raw"] < min_threshold):
            # Weak match: advance by an average gap instead of jumping to a bad candidate
            t = (line_times[-1] if line_times else 0.0) + avg_line_gap
            j = min(prev_j + primary_band // 3, N - 1)
            _log_line(f"ALIGN fallback_avg line={idx+1} score={best['score']:.1f} raw={best['raw']:.1f}")
        else:
            j = int(best["j"])
            t = max(0.0, time_at_filt[min(j, len(time_at_filt)-1)])
            # Guard against unrealistic jump: if jump > 30s and match is weak, keep modest advance
            if line_times:
                prev_t = line_times[-1]
                if (t - prev_t) > max_jump_sec and best["raw"] < good_threshold:
                    cand_j = min(prev_j + primary_band // 2, N - 1)
                    t = max(prev_t + 0.5, time_at_filt[cand_j])
                    j = cand_j
                    _log_line(f"ALIGN jump_guard applied line={idx+1} raw_score={best['raw']:.1f} prev_t={prev_t:.2f} new_t={t:.2f}")
            # Advance prev_j with inertia
            prev_j = max(prev_j, j + max(1, approx_len // 2))

        if line_times and t < line_times[-1]:
            t = line_times[-1]
        line_times.append(t)

        try:
            text_sample = ' '.join((line or '').replace('\n', ' ').split())
            if len(text_sample) > 140:
                text_sample = text_sample[:137] + '…'
        except Exception:
            text_sample = '(unavailable)'
        _log_line(f"ALIGN line={idx+1}/{len(lyrics_lines)} time={t:.3f} text={text_sample}")

    return line_times


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to a temp file and atomically replace target to avoid partial reads."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    # Write to tmp, flush, fsync, then atomic replace
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _print_devices_and_exit() -> None:
    if sd is None:
        print("sounddevice not installed. pip install sounddevice", file=sys.stderr)
        sys.exit(1)
    try:
        import json as _json
        devs = sd.query_devices()
        print("Available audio devices (index: name | in/out channels | default rate):")
        for i, d in enumerate(devs):
            print(f"  {i:2d}: {d['name']} | in:{d['max_input_channels']} out:{d['max_output_channels']} | {int(d.get('default_samplerate', 0))} Hz")
        # Also show hostapis
        try:
            ha = sd.query_hostapis()
            print("HostAPIs:")
            for i, h in enumerate(ha):
                print(f"  {i:2d}: {h['name']} (devices: {h.get('device_count', '?')})")
        except Exception:
            pass
    except Exception as e:
        print("Failed to query devices:", e, file=sys.stderr)
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description="Align song lyrics to times from mic/whisper")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--song-id", help="Song ID (JSON in static/data/<id>.json)")
    src.add_argument("--current", action="store_true", help="Use /data/now_playing.json from the running app")
    ap.add_argument("--duration", type=float, default=60.0, help="Record duration (seconds) if using mic")
    ap.add_argument("--input", help="Optional path to an existing WAV/MP3/FLAC instead of mic")
    ap.add_argument("--model", default="small", help="Faster-Whisper model size (tiny, base, small, medium, large)")
    ap.add_argument("--whisper-device", default=os.getenv("WHISPER_DEVICE", "auto"), choices=["auto","cpu","cuda"], help="Device for Whisper (default: auto)")
    ap.add_argument("--compute-type", default=os.getenv("WHISPER_COMPUTE", None), help="Optional compute type for Whisper (e.g., int8, int8_float16, float16)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write file; just print first few times")
    ap.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    ap.add_argument("--device", help="Input device index or substring name", default=None)
    ap.add_argument("--samplerate", type=int, default=None, help="Override input samplerate (Hz)")
    ap.add_argument("--channels", type=int, default=1, help="Mic channels (1=mono,2=stereo)")
    ap.add_argument("--hostapi", type=int, default=None, help="Optional HostAPI index (see --list-devices)")
    args = ap.parse_args()

    if args.list_devices:
        _print_devices_and_exit()

    # If not listing devices, require a song selector
    if not args.song_id and not args.current:
        ap.error("one of the arguments --song-id --current is required")

    if args.current:
        state_path = ROOT / "data" / "now_playing.json"
        if not state_path.exists():
            print("No now_playing.json found. Open /app, select a song, and try again.", file=sys.stderr)
            sys.exit(1)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        sid = state.get("song_id")
        if not sid:
            print("now_playing.json missing song_id", file=sys.stderr)
            sys.exit(1)
        song_path = DATA_DIR / f"{sid}.json"
        print(f"Using current song_id from app: {sid}")
        _log_line(f"JOB start pid={os.getpid()} song_id={sid} model={args.model} duration={args.duration}")
    else:
        song_path = DATA_DIR / f"{args.song_id}.json"
        sid = Path(song_path).stem
        _log_line(f"JOB start pid={os.getpid()} song_id={sid} model={args.model} duration={args.duration}")
    if not song_path.exists():
        print(f"Song not found: {song_path}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(song_path.read_text(encoding="utf-8"))
    lines = data.get("lines") or []
    if not lines:
        print("Song JSON has no lines", file=sys.stderr)
        sys.exit(1)
    lyric_texts = [ln.get("text") or "" for ln in lines]

    # Get audio
    if args.input:
        audio_path = Path(args.input)
        if not audio_path.exists():
            print(f"Input not found: {audio_path}", file=sys.stderr)
            sys.exit(1)
    else:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CAPTURE_DIR / f"capture_{int(time.time())}.wav"
        # Resolve device argument: allow integer index or substring matching
        device = None
        if args.device is not None:
            if args.device.isdigit():
                device = int(args.device)
            else:
                # substring search over device names
                try:
                    devs = sd.query_devices()
                    for i, d in enumerate(devs):
                        if args.device.lower() in str(d.get('name','')).lower() and d.get('max_input_channels',0)>0:
                            device = i
                            break
                except Exception:
                    device = None
        record_to_wav(tmp, duration=args.duration, samplerate=args.samplerate, channels=args.channels, device=device, hostapi=args.hostapi)
        audio_path = tmp

    # Transcribe → words
    words = transcribe_words(audio_path, model_size=args.model, whisper_device=args.whisper_device, compute_type=args.compute_type)
    if not words:
        sys.exit(2)

    # Align → line start times (log periodic progress)
    line_times = align_lines(lyric_texts, words)

    # Update JSON
    for i, t in enumerate(line_times):
        try:
            lines[i]["time"] = float(round(t, 3))
        except Exception:
            pass

    if args.dry_run:
        print("First 10 line times:")
        for i, t in list(enumerate(line_times))[:10]:
            print(f"  {i+1:02d} {t:8.3f}s  |  {lyric_texts[i][:60]}")
        print("(dry run — no file written)")
        return

    _write_job({"phase": "writing"})
    atomic_write_json(song_path, data)
    _log_line(f"WRITE json path={song_path} lines={len(lines)} model={args.model}")
    # Log final full mapping (line -> time)
    _log_line(f"SUMMARY begin lines={len(lines)} song={sid}")
    for i, ln in enumerate(lines):
        try:
            tval = float(ln.get('time', 0.0))
        except Exception:
            tval = 0.0
        try:
            txt = ' '.join((ln.get('text','') or '').replace('\n',' ').split())
            if len(txt) > 140:
                txt = txt[:137] + '…'
        except Exception:
            txt = '(unavailable)'
        _log_line(f"FINAL line={i+1}/{len(lines)} time={tval:.3f} text={txt}")
    _log_line("SUMMARY end")
    _write_job({"phase": "done", "finished_at": int(time.time())})
    print(f"Wrote updated timestamps to {song_path}")


if __name__ == "__main__":
    main()
