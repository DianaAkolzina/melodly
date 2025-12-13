import re
import unicodedata
from pathlib import Path
from typing import Iterable, List


def slugify(text: str) -> str:
    """Create a filesystem-safe ASCII slug from arbitrary text."""
    # Normalize accents/diacritics, drop non-ascii
    normalized = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    # Lower, replace non-alnum with hyphens
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)
    return slug or "untitled"


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def estimate_line_timestamps(
    lines: Iterable[str],
    total_duration: float | None = None,
    words_per_second: float = 2.6,
    min_line_duration: float = 0.9,
    min_pause: float = 0.3,
    max_pause: float = 1.2,
) -> List[float]:
    """Estimate monotonically increasing start times for lyric lines.

    The heuristic distributes time based on word counts while preserving
    optional gaps between lines. Returns start times in seconds.
    """
    filtered = [ln.strip() for ln in lines if ln and ln.strip()]
    if not filtered:
        return []

    word_counts = [max(len(re.findall(r"\w+", ln)), 1) for ln in filtered]
    total_words = sum(word_counts)

    if total_duration is None or total_duration <= 0:
        estimated = total_words / max(words_per_second, 0.1)
        baseline = len(filtered) * (min_line_duration + min_pause)
        total_duration = max(estimated, baseline)
    else:
        total_duration = max(total_duration, len(filtered) * min_line_duration)

    pause = 0.0
    if len(filtered) > 1:
        pause = min(max_pause, max(min_pause, total_duration * 0.05 / (len(filtered) - 1)))

    content_time = total_duration - pause * (len(filtered) - 1)
    if content_time < len(filtered) * min_line_duration:
        content_time = len(filtered) * min_line_duration
        total_duration = content_time + pause * (len(filtered) - 1)

    if total_words == 0:
        durations = [min_line_duration] * len(filtered)
    else:
        proportions = [wc / total_words for wc in word_counts]
        durations = [max(min_line_duration, content_time * p) for p in proportions]

    def shrink_to_target(values: List[float], target: float, minimum: float) -> List[float]:
        total = sum(values)
        if total <= target:
            return values
        adjustable = [max(v - minimum, 0.0) for v in values]
        adjustable_sum = sum(adjustable)
        if adjustable_sum <= 0:
            return values
        factor = min(1.0, (total - target) / adjustable_sum)
        adjusted: List[float] = []
        for v, slack in zip(values, adjustable):
            reduction = slack * factor
            adjusted.append(max(minimum, v - reduction))
        return adjusted

    durations = shrink_to_target(durations, content_time, min_line_duration)

    timestamps: List[float] = []
    current = 0.0
    for idx, dur in enumerate(durations):
        timestamps.append(round(current, 3))
        current += dur
        if idx < len(durations) - 1:
            current += pause

    return timestamps
