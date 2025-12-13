# Multilingual Songs: Lyrics, Translation, and Grammar Packages

This project scaffolds a small pipeline to:

1) Curate 20 popular songs on Spotify across Spanish, French, Irish Gaelic, Latvian, and Russian (4 each).
2) Fetch lyrics from Musixmatch (where available via API).
3) Generate JSON packages with English translations, glossary, grammar notes, and approximate per-line timestamps via OpenAI or Ollama.

Important notes:
- Musixmatch free API typically returns only partial lyrics and includes a disclaimer; the script strips the disclaimer but cannot obtain full lyrics without a license.
- Lyrics are copyrighted. This repo stores the lyrics locally on your machine for study; do not re-publish.

## Setup

1) Python 3.10+ recommended

2) Install deps:

   pip install -r requirements.txt

3) Environment variables: copy `.env.example` to `.env` and fill in keys:

   - `MUSIXMATCH_API_KEY`: Musixmatch API key
   - `ENGINE`: `openai` (default) or `ollama`
   - `OPENAI_API_KEY`: if `ENGINE=openai`
   - `OPENAI_MODEL`: Defaults to `gpt-4o-mini`
   - `OLLAMA_HOST`: Defaults to `http://localhost:11434`
   - `OLLAMA_MODEL`: Defaults to `llama3`

## Data inputs

- `data/songs.yaml` contains the curated list with language codes and artists.

## Fetch lyrics

   python scripts/fetch_lyrics.py

Outputs per song to `data/raw_lyrics/<lang>/<slug>.txt` and metadata to `.meta.json`.

## Translate and annotate

   ENGINE=ollama OLLAMA_MODEL=llama3 python scripts/translate_and_annotate.py

Outputs JSON to `data/processed/<lang>/<slug>.json` following `schema/song_package.schema.json` (now includes `timestamps`).

## Languages

- Gaelic here is Irish Gaelic (`ga`). If you want Scottish Gaelic instead, say the word and we can swap titles.

## Caveats

- Track search is best-effort and uses title + artist; adjust `data/songs.yaml` if a different artist or edition retrieves better lyrics.
- API rate limiting is handled gently but heavy usage may still be throttled.

## Optional safety check: mixed-language filter

Set `SKIP_MIXED_LANG=1` to auto-skip songs whose lyrics contain a significant portion of non-target-language lines (LLM-estimated). Threshold defaults to 25%.
