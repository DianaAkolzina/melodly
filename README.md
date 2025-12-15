Meelange — Lyrics + Translation + Grammar

Overview

- Small Flask web app (Meelange) that demos learning a language through singing.
- Includes a French public‑domain song ("Au clair de la lune") with:
  - Time-synced lyrics
  - Side-by-side English translation (default)
  - Click a lyric line to see a short grammar note
- Audio is optional; demo uses simulated timing if no audio file is present.

Quick Start

1) Create a virtual env and install deps

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2) Run the app

   python app.py

3) Open in browser

   http://127.0.0.1:5000/

Optional: Add audio

- Best place for MP3s: static/audio/
- Expected filenames (you can change in each song JSON):
  - static/audio/au_clair_de_la_lune.mp3
  - static/audio/frere_jacques.mp3
  - static/audio/a_la_claire_fontaine.mp3
- The UI uses each song’s audio path from JSON; if missing, it simulates timing.

Project Structure

- app.py — Flask server and minimal APIs
- templates/index.html — Main page shell
- static/app.js — Frontend logic (sync, playback, grammar)
- static/styles.css — Minimal styling
- static/data/au_clair_de_la_lune.json — Song data (lyrics + translation + grammar + timestamps)
- static/data/frere_jacques.json — Additional song
- static/data/a_la_claire_fontaine.json — Additional song
- static/data/sur_le_pont_davignon.json — Additional song (full lyrics)
- static/data/alouette.json — Additional song (full lyrics)
- static/audio/ — Optional audio directory (empty by default)

Notes

- This is a demo: translations and grammar notes are simplified.
- You can add more songs by copying the JSON pattern.

Translation models (prefetch for offline/faster runs)

- The translation filler prefers per-language Marian MT models (fr/es/pt/it/de/lv/ga/gd/ru → en) with `opus-mt-mul-en` as fallback.
- To cache them locally: `python scripts/download_translation_models.py`
- You can override the cache dir if needed: `python scripts/download_translation_models.py --cache-dir /path/to/cache`

Songs Catalog

- Au clair de la lune (A1): Classic children’s song; simple imperatives and negation; timestamps aligned to common tempo; Spotify search uses title + artist "Traditional".
- Frère Jacques (A1): Repetitive structure; present tense and imperatives; timestamps approximate to nursery standard.
- À la claire fontaine (A2): Folk song with passé composé and futur; refrain repeats; timestamps at ~3–4s per line.
- Sur le pont d'Avignon (A1): Full lyrics with multiple trade/role verses; refrain every 2 lines; timestamps ~2.5s per line.
- Alouette, gentille alouette (A1): Cumulative refrain building body parts; timestamps ~2s per line, longer at refrains.

Timing and Spotify

- Playback uses Spotify. On song select, the app auto-searches Spotify by song title/artist and selects the top result; you can override by pasting a `spotify:track:` URI or choosing a different search result.
- Timestamps are matched to common recordings. If the highlight drifts with your chosen track/version, pick a closer match from the dropdown. Fine-tuning per-line timings can be done by editing the song JSON times.

Add Your Own Song

- Copy a JSON in `static/data/`, change `id`, `title`, `artist`, `level`, and fill `lines` with objects: `{ time, text, translation, grammar }`.
- Keep `from_language` = original (e.g., `fr`) and `to_language` = your learning target (e.g., `en`). The UI treats the original as the learned language.
- Prefer public-domain lyrics. Provide one logical phrase per line with a time in seconds. The app highlights the last line whose `time` is <= current playback time.

Spotify Full-Track Playback (Optional)

To play full tracks via Spotify in the browser (Web Playback SDK), set these env vars and use the auth flow at /auth/spotify/login:

- SPOTIFY_CLIENT_ID=your_client_id
- SPOTIFY_CLIENT_SECRET=your_client_secret
- SPOTIFY_REDIRECT_URI=http://localhost:5000/callback/spotify
- SPOTIFY_SCOPES="streaming user-read-email user-read-private user-read-playback-state user-modify-playback-state user-read-currently-playing user-library-read"
- FLASK_SECRET_KEY=some_random_string

Spotify Redirect URI (current guidance)

- Spotify is deprecating insecure redirects. For local loopback they recommend using HTTP with a loopback address (127.0.0.1) and allow dynamic ports.
- Set both places to the exact same value:
  - Spotify dashboard Redirect URIs: `http://127.0.0.1:5000/callback/spotify`
  - `.env`: `SPOTIFY_REDIRECT_URI=http://127.0.0.1:5000/callback/spotify`
- Start the app; it binds to `127.0.0.1:5000` when `USE_HTTPS=false`.
- Open `http://127.0.0.1:5000/` and click Connect Spotify.

Optional HTTPS

- If you prefer HTTPS locally, set `USE_HTTPS=true` and switch both `.env` and the dashboard to `https://localhost:5000/callback/spotify`. You may need a trusted cert (e.g., mkcert) to avoid browser warnings.

Flow

- Visit http://localhost:5000/auth/spotify/login to sign in.
- Frontend (to be wired) will provide the SDK device_id to /api/spotify/transfer to make the browser the active device.
- Use /api/spotify/play with {"uris": ["spotify:track:..."]} to start playback, or /api/spotify/pause to pause.
- Get your access token for the SDK via /api/spotify/token (the backend refreshes as needed).

Notes/limits: Spotify disallows downloading audio; playback streams via Spotify only. You need a Premium account for the Web Playback SDK.

Fetching Lyrics via Musixmatch + Translating with Ollama

- This repo includes `scripts/fetch_musixmatch_songs.py` to populate `static/data` with 20 songs (4 each): Gaelic, Latvian, Basque, Spanish, French.
- The script fetches new/popular songs with lyrics from Musixmatch, asks a local Ollama model to translate and add brief grammar notes per line, and saves JSONs matching the app schema.

Setup

- Set your Musixmatch API key without committing it:
  - `export MUSIXMATCH_API_KEY=your_key` (or put it in `.env`)
- Ensure Ollama is running locally at `http://localhost:11434` (default). Optionally set `OLLAMA_MODEL` (default `llama3`).

Run

- `python scripts/fetch_musixmatch_songs.py`

Details

- The script tries 4 songs per language; on failure (no lyrics/restricted), it retries with another track.
- Gaelic uses codes `gd` then `ga`.
- Timestamps are synthetic (3s per line). Update `audio`/`duration` if you have actual media.
- Files are saved as `static/data/<lang>_<title_slug>.json`.
- The script removes common Musixmatch footer/disclaimer lines; respect licensing and Musixmatch terms.
