from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import subprocess
import signal
import sys

import requests
from langdetect import detect, DetectorFactory
from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    render_template,
    abort,
    redirect,
    request,
    session,
    url_for,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "static" / "data"
STATE_DIR = BASE_DIR / "data"
STATE_DIR.mkdir(exist_ok=True)
NOW_PLAYING_FILE = STATE_DIR / "now_playing.json"
TS_JOB_FILE = STATE_DIR / "timestamps_job.json"
TS_LOG_FILE = STATE_DIR / "timestamps.log"
DetectorFactory.seed = 0  # stable langdetect results
AUTO_ALIGN_ON_PLAY = os.getenv("AUTO_ALIGN_ON_PLAY", "0").strip().lower() in {"1", "true", "yes", "on"}
AUTO_ALIGN_MODEL = os.getenv("AUTO_ALIGN_MODEL", "large")

# Constants
DEFAULT_SPOTIFY_SCOPES = (
    "streaming user-read-email user-read-private user-read-playback-state "
    "user-modify-playback-state user-read-currently-playing user-library-read"
)
TOKEN_BUFFER_SECONDS = 60  # Buffer before token expiry
REQUEST_TIMEOUT = 15


class SpotifyError(Exception):
    """Custom exception for Spotify-related errors"""
    pass


class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass


def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get environment variable with optional validation"""
    val = os.getenv(name, default)
    if required and not val:
        raise ConfigError(f"Required environment variable {name} not set")
    return val


def _spotify_creds() -> Tuple[str, str, str, str]:
    """Get Spotify credentials from environment"""
    client_id = _get_env("SPOTIFY_CLIENT_ID", required=True)
    client_secret = _get_env("SPOTIFY_CLIENT_SECRET", required=True)
    redirect_uri = _get_env("SPOTIFY_REDIRECT_URI", "http://localhost:5000/callback/spotify")
    scopes = _get_env("SPOTIFY_SCOPES", DEFAULT_SPOTIFY_SCOPES)
    
    return client_id, client_secret, redirect_uri, scopes


def _now() -> int:
    """Get current timestamp"""
    return int(time.time())


def _validate_json_payload(data: Dict[str, Any], required_fields: list[str]) -> None:
    """Validate JSON payload has required fields"""
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.secret_key = _get_env("FLASK_SECRET_KEY", "dev-secret-not-for-production")
    
    if not app.secret_key or app.secret_key == "dev-secret-not-for-production":
        logger.warning("Using default secret key - set FLASK_SECRET_KEY for production")

    @app.route("/")
    def home():
        """Serve marketing home page"""
        return render_template("home.html")

    @app.route("/app")
    def app_page():
        """Serve main application page"""
        return render_template("index.html")

    @app.route("/pricing")
    def pricing():
        return render_template("pricing.html")

    @app.route("/contact")
    def contact():
        return render_template("contact.html")

    @app.route("/api/songs")
    def list_songs():
        """List available songs from JSON files"""
        try:
            songs = []
            if not DATA_DIR.exists():
                logger.warning(f"Data directory {DATA_DIR} does not exist")
                return jsonify({"songs": []})
                
            for p in DATA_DIR.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    
                    # Validate required fields
                    required_fields = ["id", "title", "artist"]
                    if all(field in data for field in required_fields):
                        songs.append({
                            "id": data.get("id"),
                            "title": data.get("title"),
                            "artist": data.get("artist"),
                            "from_language": data.get("from_language"),
                            "to_language": data.get("to_language"),
                            "level": data.get("level"),
                            "is_excerpt": data.get("is_excerpt", False),
                            "is_placeholder": data.get("is_placeholder", False),
                        })
                    else:
                        logger.warning(f"Invalid song data in {p.name}: missing required fields")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in {p.name}")
                except Exception as e:
                    logger.error(f"Error processing {p.name}: {e}")
            resp = jsonify({"songs": songs})
            # Always deliver a fresh list; avoid cached dropdown data
            resp.headers["Cache-Control"] = "no-store"
            logger.info(f"/api/songs returned {len(songs)} items")
            return resp
        except Exception as e:
            logger.error(f"Error listing songs: {e}")
            return jsonify({"error": "Failed to load songs"}), 500

    @app.route("/api/song/<song_id>")
    def get_song(song_id: str):
        """Get detailed song information"""
        # Sanitize song_id to prevent path traversal
        if not song_id.replace("-", "").replace("_", "").isalnum():
            abort(400)
            
        p = DATA_DIR / f"{song_id}.json"
        if not p.exists() or not p.is_file():
            abort(404)
            
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            resp = jsonify(data)
            # Ensure clients always fetch fresh JSON (no caching)
            resp.headers["Cache-Control"] = "no-store"
            return resp
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {song_id}.json")
            abort(500)
        except Exception as e:
            logger.error(f"Error loading song {song_id}: {e}")
            abort(500)

    @app.route("/api/now-playing", methods=["GET", "POST"])
    def now_playing():
        """Get or set the current song selection (front-end hint for tooling)."""
        try:
            if request.method == "GET":
                if NOW_PLAYING_FILE.exists():
                    return jsonify(json.loads(NOW_PLAYING_FILE.read_text(encoding="utf-8")))
                return jsonify({"song_id": None})

            # POST
            payload = request.get_json(silent=True) or {}
            song_id = (payload.get("song_id") or "").strip()
            title = (payload.get("title") or "").strip()
            artist = (payload.get("artist") or "").strip()
            if not song_id:
                return jsonify({"error": "song_id_required"}), 400
            NOW_PLAYING_FILE.write_text(json.dumps({
                "song_id": song_id,
                "title": title,
                "artist": artist,
                "ts": int(time.time()),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            # Auto-start mic alignment for current song unless explicitly skipped
            auto_align_requested = bool(payload.get("auto_align", True))
            if AUTO_ALIGN_ON_PLAY and auto_align_requested:
                try:
                    song_path = DATA_DIR / f"{song_id}.json"
                    duration = 120
                    if song_path.exists():
                        song_data = json.loads(song_path.read_text(encoding="utf-8"))
                        dur_field = song_data.get("duration")
                        if isinstance(dur_field, (int, float)) and dur_field > 5:
                            duration = int(dur_field)
                        else:
                            lines = song_data.get("lines") or []
                            duration = max(60, int(len(lines) * 3.0))
                    _launch_timestamp_job(song_id, duration=duration, model=AUTO_ALIGN_MODEL)
                except Exception as e:
                    logger.error(f"auto align failed to start: {e}")
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"now_playing error: {e}")
            return jsonify({"error": "server_error"}), 500

    # ---------- Mic Whisper auto timestamp job control ----------

    def _read_job() -> Optional[Dict[str, Any]]:
        if TS_JOB_FILE.exists():
            try:
                return json.loads(TS_JOB_FILE.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _write_job(info: Dict[str, Any]) -> None:
        TS_JOB_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _stop_job_silent():
        job = _read_job()
        if not job:
            return False
        pid = int(job.get("pid") or 0)
        if pid and _pid_alive(pid):
            try:
                # Try to terminate process group first
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
        try:
            TS_JOB_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        return True

    def _launch_timestamp_job(
        song_id: str,
        duration: int,
        model: str = "small",
        device: Optional[str] = None,
        samplerate: Optional[int] = None,
        channels: Optional[int] = None,
        hostapi: Optional[int] = None,
        capture_source: str = "loopback",
    ) -> Optional[Dict[str, Any]]:
        """Start mic_whisper_align.py in the background for the current song."""
        script = BASE_DIR / "scripts" / "mic_whisper_align.py"
        if not script.exists():
            return None
        _stop_job_silent()
        cmd = [sys.executable, str(script), "--current", "--duration", str(duration), "--model", model, "--capture-source", capture_source]
        if device is not None:
            cmd += ["--device", str(device)]
        if samplerate is not None:
            cmd += ["--samplerate", str(int(samplerate))]
        if channels is not None:
            cmd += ["--channels", str(int(channels))]
        if hostapi is not None:
            cmd += ["--hostapi", str(int(hostapi))]
        proc = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        info = {
            "pid": proc.pid,
            "song_id": song_id,
            "duration_sec": duration,
            "model": model,
            "started_at": _now(),
        }
        _write_job(info)
        try:
            TS_LOG_FILE.parent.mkdir(exist_ok=True)
            with TS_LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(
                    f"{info['started_at']} START pid={proc.pid} song_id={song_id} model={model} duration={duration}"
                    f" device={device} samplerate={samplerate} channels={channels} hostapi={hostapi} capture_source={capture_source}\n"
                )
        except Exception:
            pass
        return info

    @app.route("/api/timestamps/status")
    def timestamps_status():
        job = _read_job() or {}
        pid = int(job.get("pid") or 0)
        job["running"] = bool(pid and _pid_alive(pid))
        return jsonify(job)

    @app.route("/api/timestamps/log")
    def timestamps_log_tail():
        try:
            count = int(request.args.get("lines", 100))
        except Exception:
            count = 100
        count = max(1, min(1000, count))
        lines: list[str] = []
        try:
            if TS_LOG_FILE.exists():
                data = TS_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
                lines = data[-count:]
        except Exception as e:
            logger.error(f"failed to read log: {e}")
        return jsonify({"lines": lines})

    @app.route("/api/timestamps/cancel", methods=["POST"]) 
    def timestamps_cancel():
        job = _read_job() or {}
        phase = (job.get("phase") or "").lower()
        # If the job is already in transcribing/aligning/writing, let it finish to ensure JSON is written
        if phase in ("transcribing", "aligning", "writing"):
            try:
                TS_LOG_FILE.parent.mkdir(exist_ok=True)
                with TS_LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(f"{_now()} CANCEL requested; stopped=False (finalizing {phase})\n")
            except Exception:
                pass
            return jsonify({"stopped": False, "reason": f"finalizing_{phase}"})

        stopped = _stop_job_silent()
        try:
            TS_LOG_FILE.parent.mkdir(exist_ok=True)
            with TS_LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(f"{_now()} CANCEL requested; stopped={stopped}\n")
        except Exception:
            pass
        return jsonify({"stopped": stopped})

    @app.route("/api/timestamps/auto-start", methods=["POST"]) 
    def timestamps_auto_start():
        try:
            payload = request.get_json(silent=True) or {}
            duration = max(1, int(float(payload.get("duration_sec") or 0)))
            model = (payload.get("model") or "small").strip()
            device = payload.get("device")  # index or name substring
            samplerate = payload.get("samplerate")
            channels = payload.get("channels")
            hostapi = payload.get("hostapi")
            capture_source = (payload.get("capture_source") or "loopback").lower()
            if capture_source not in ("loopback", "mic", "auto"):
                capture_source = "loopback"

            # determine current song id
            if NOW_PLAYING_FILE.exists():
                state = json.loads(NOW_PLAYING_FILE.read_text(encoding="utf-8"))
                song_id = state.get("song_id")
            else:
                song_id = None
            if not song_id:
                return jsonify({"error": "no_current_song"}), 400

            info = _launch_timestamp_job(
                song_id,
                duration=duration,
                model=model,
                device=device,
                samplerate=samplerate,
                channels=channels,
                hostapi=hostapi,
                capture_source=capture_source,
            )
            if not info:
                return jsonify({"error": "script_missing"}), 500
            return jsonify({"started": True, **info})
        except Exception as e:
            logger.error(f"auto-start error: {e}")
            return jsonify({"error": "start_failed"}), 500

    @app.route("/api/song/<song_id>/timestamps", methods=["POST"])
    def update_song_timestamps(song_id: str):
        """Update one or more line timestamps for a song.

        Accepts either:
        - {"updates": [{"index": int, "time": float}, ...]}
        - {"times": [float, ...]}  # full-length list aligned to lines
        """
        # Sanitize song_id
        if not song_id.replace("-", "").replace("_", "").isalnum():
            abort(400)

        p = DATA_DIR / f"{song_id}.json"
        if not p.exists() or not p.is_file():
            abort(404)

        try:
            payload = request.get_json(force=True, silent=False) or {}
            data = json.loads(p.read_text(encoding="utf-8"))
            lines: List[Dict[str, Any]] = data.get("lines") or []
            if not isinstance(lines, list) or not lines:
                return jsonify({"error": "no_lines"}), 400

            if "updates" in payload:
                updates = payload.get("updates") or []
                if not isinstance(updates, list):
                    return jsonify({"error": "invalid_updates"}), 400
                for u in updates:
                    try:
                        idx = int(u.get("index"))
                        t = float(u.get("time"))
                    except Exception:
                        return jsonify({"error": "invalid_update_item"}), 400
                    if not (0 <= idx < len(lines)):
                        return jsonify({"error": "index_out_of_range", "index": idx}), 400
                    lines[idx]["time"] = max(0.0, float(t))
            elif "times" in payload:
                times = payload.get("times")
                if not isinstance(times, list) or len(times) != len(lines):
                    return jsonify({"error": "invalid_times"}), 400
                for i, t in enumerate(times):
                    try:
                        lines[i]["time"] = max(0.0, float(t))
                    except Exception:
                        return jsonify({"error": "invalid_time_value", "index": i}), 400
            else:
                return jsonify({"error": "no_updates_provided"}), 400

            # Ensure times are non-decreasing
            prev = 0.0
            for i in range(len(lines)):
                try:
                    cur = float(lines[i].get("time", 0.0))
                except Exception:
                    cur = 0.0
                if cur < prev:
                    lines[i]["time"] = prev
                prev = float(lines[i]["time"])

            data["lines"] = lines
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Error updating timestamps for {song_id}: {e}")
            return jsonify({"error": "failed_to_update"}), 500

    # -------- Spotify OAuth + Playback Backend --------

    @app.route("/auth/spotify/login")
    def spotify_login():
        """Initiate Spotify OAuth flow"""
        try:
            client_id, _, redirect_uri, scopes = _spotify_creds()
            
            state = os.urandom(16).hex()
            session["sp_oauth_state"] = state
            
            params = {
                "client_id": client_id,
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "scope": scopes,
                "state": state,
                "show_dialog": "false",
            }
            
            from urllib.parse import urlencode
            auth_url = "https://accounts.spotify.com/authorize?" + urlencode(params)
            
            logger.info(f"Redirecting to Spotify OAuth with redirect_uri={redirect_uri}")
            return redirect(auth_url)
            
        except ConfigError as e:
            logger.error(f"Configuration error: {e}")
            return jsonify({"error": "Server configuration error"}), 500
        except Exception as e:
            logger.error(f"Error initiating Spotify login: {e}")
            return jsonify({"error": "Failed to initiate Spotify login"}), 500

    @app.route("/callback/spotify")
    def spotify_callback():
        """Handle Spotify OAuth callback"""
        try:
            error = request.args.get("error")
            if error:
                logger.error(f"Spotify authorization error: {error}")
                return jsonify({"error": f"Spotify authorization failed: {error}"}), 400

            code = request.args.get("code")
            state = request.args.get("state")
            
            if not code:
                return jsonify({"error": "Authorization code not received"}), 400
                
            # Validate state parameter (CSRF protection)
            if state != session.get("sp_oauth_state"):
                logger.warning("OAuth state mismatch - possible CSRF attack")
                return jsonify({"error": "Invalid state parameter"}), 400

            client_id, client_secret, redirect_uri, _ = _spotify_creds()

            # Exchange authorization code for access token
            token_url = "https://accounts.spotify.com/api/token"
            data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            }
            
            auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
            headers = {
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            
            resp = requests.post(token_url, data=data, headers=headers, timeout=REQUEST_TIMEOUT)
            
            if resp.status_code != 200:
                logger.error(f"Token exchange failed: {resp.status_code} - {resp.text}")
                return jsonify({"error": "Failed to exchange authorization code"}), 400

            payload = resp.json()
            access_token = payload.get("access_token")
            refresh_token = payload.get("refresh_token")
            expires_in = payload.get("expires_in", 3600)

            if not access_token:
                return jsonify({"error": "No access token received"}), 400

            # Store tokens in session
            session["sp_access_token"] = access_token
            session["sp_refresh_token"] = refresh_token
            session["sp_expires_at"] = _now() + int(expires_in) - TOKEN_BUFFER_SECONDS

            # Clear OAuth state
            session.pop("sp_oauth_state", None)
            
            logger.info("Successfully authenticated with Spotify")
            return redirect(url_for("app_page"))
            
        except ConfigError as e:
            logger.error(f"Configuration error: {e}")
            return jsonify({"error": "Server configuration error"}), 500
        except Exception as e:
            logger.error(f"Error in Spotify callback: {e}")
            return jsonify({"error": "Authentication failed"}), 500

    def _refresh_access_token() -> bool:
        """Refresh Spotify access token using refresh token"""
        try:
            refresh_token = session.get("sp_refresh_token")
            if not refresh_token:
                return False
                
            client_id, client_secret, _, _ = _spotify_creds()

            token_url = "https://accounts.spotify.com/api/token"
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }
            
            auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
            headers = {
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            
            resp = requests.post(token_url, data=data, headers=headers, timeout=REQUEST_TIMEOUT)
            
            if resp.status_code != 200:
                logger.error(f"Token refresh failed: {resp.status_code}")
                return False

            payload = resp.json()
            access_token = payload.get("access_token")
            expires_in = payload.get("expires_in", 3600)
            
            if access_token:
                session["sp_access_token"] = access_token
                session["sp_expires_at"] = _now() + int(expires_in) - TOKEN_BUFFER_SECONDS
                
                # Update refresh token if provided
                if payload.get("refresh_token"):
                    session["sp_refresh_token"] = payload.get("refresh_token")
                    
                logger.info("Successfully refreshed Spotify access token")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error refreshing access token: {e}")
            return False

    def _ensure_access_token() -> Optional[str]:
        """Ensure we have a valid access token"""
        token = session.get("sp_access_token")
        expires_at = session.get("sp_expires_at", 0)
        
        # Check if current token is still valid
        if token and _now() < int(expires_at):
            return token
            
        # Try to refresh token
        if _refresh_access_token():
            return session.get("sp_access_token")
            
        return None

    def _make_spotify_request(method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make authenticated request to Spotify API"""
        token = _ensure_access_token()
        if not token:
            raise SpotifyError("No valid access token")
            
        headers = kwargs.pop('headers', {})
        headers["Authorization"] = f"Bearer {token}"
        
        url = f"https://api.spotify.com/v1{endpoint}"
        response = requests.request(method, url, headers=headers, timeout=REQUEST_TIMEOUT, **kwargs)
        
        if response.status_code == 401:
            raise SpotifyError("Authentication failed")
            
        return response

    # -------- Helpers: Language detection for tracks --------

    def _mm_detect_language(title: str, artist: str) -> Optional[str]:
        """Try to detect track language using Musixmatch lyrics metadata."""
        api_key = _get_env("MUSIXMATCH_API_KEY")
        if not api_key:
            return None
        base = "https://api.musixmatch.com/ws/1.1"
        try:
            # Search track
            params = {
                "q_track": title,
                "q_artist": artist,
                "f_has_lyrics": 1,
                "page_size": 1,
                "s_track_rating": "desc",
            }
            r = requests.get(f"{base}/track.search", params={**params, "apikey": api_key}, timeout=10)
            r.raise_for_status()
            track_list = r.json().get("message", {}).get("body", {}).get("track_list", [])
            if not track_list:
                return None
            track_id = track_list[0].get("track", {}).get("track_id")
            if not track_id:
                return None
            # Fetch lyrics to read language
            r2 = requests.get(f"{base}/track.lyrics.get", params={"track_id": track_id, "apikey": api_key}, timeout=10)
            r2.raise_for_status()
            lyr = r2.json().get("message", {}).get("body", {}).get("lyrics", {})
            lang = lyr.get("lyrics_language") or lyr.get("lyrics_lang")
            if isinstance(lang, str) and len(lang) in (2, 3):
                return lang.lower()
        except Exception:
            return None
        return None

    def _heuristic_detect_language(title: str, artist: str) -> Optional[str]:
        text = f"{title} {artist}".strip()
        if not text:
            return None
        try:
            code = detect(text)
            return code
        except Exception:
            return None

    def detect_language_for_track(title: str, artist: str) -> Optional[str]:
        lang = _mm_detect_language(title, artist)
        if lang:
            return lang
        return _heuristic_detect_language(title, artist)

    @app.route("/api/spotify/token")
    def spotify_token():
        """Get current Spotify access token"""
        token = _ensure_access_token()
        if not token:
            return jsonify({"error": "unauthorized"}), 401
        return jsonify({
            "access_token": token, 
            "expires_at": session.get("sp_expires_at")
        })

    @app.route("/api/spotify/me")
    def spotify_me():
        """Get current user's Spotify profile"""
        try:
            resp = _make_spotify_request("GET", "/me")
            return jsonify(resp.json()), resp.status_code
        except SpotifyError:
            return jsonify({"error": "unauthorized"}), 401
        except Exception as e:
            logger.error(f"Error getting Spotify profile: {e}")
            return jsonify({"error": "Failed to get profile"}), 500

    @app.route("/api/spotify/search-and-play", methods=["POST"])
    def spotify_search_and_play():
        """Search for a song and automatically play it with enhanced track info"""
        try:
            body = request.get_json(silent=True) or {}
            
            # Validate required fields
            title = body.get("title", "").strip()
            artist = body.get("artist", "").strip()
            
            if not title or not artist:
                return jsonify({"error": "Both title and artist are required"}), 400
                
            # Search for the song with multiple attempts for better matching
            search_queries = [
                f'track:"{title}" artist:"{artist}"',  # Most specific
                f'"{title}" artist:"{artist}"',  # Exact match with quotes
                f"{title} {artist}",  # Simple combination
                f'track:"{title}"',  # Just track name if artist fails
                f"{title}"  # Last resort
            ]
            
            track = None
            all_results = []
            
            for search_query in search_queries:
                resp = _make_spotify_request(
                    "GET", 
                    "/search",
                    params={"q": search_query, "type": "track", "limit": 10}  # Get more results
                )
                
                if resp.status_code == 200:
                    search_data = resp.json()
                    tracks = search_data.get("tracks", {}).get("items", [])
                    all_results.extend(tracks)
                    
                    if tracks:
                        # Smart matching logic
                        for t in tracks:
                            track_name = t.get("name", "").lower().strip()
                            track_artists = [a.get("name", "").lower().strip() for a in t.get("artists", [])]
                            
                            # Filter out likely instrumental/karaoke versions
                            if any(keyword in track_name for keyword in ['instrumental', 'karaoke', 'backing track', 'minus one']):
                                continue
                                
                            # Exact title and artist match
                            if (track_name == title.lower().strip() and 
                                any(artist.lower().strip() in track_artist for track_artist in track_artists)):
                                track = t
                                break
                                
                            # Close title match with right artist
                            if (track_name.replace("-", "").replace(":", "") == title.lower().replace("-", "").replace(":", "") and
                                any(artist.lower().strip() in track_artist for track_artist in track_artists)):
                                track = t
                                break
                        
                        if track:
                            break
                        
                        # If no perfect match, take first non-instrumental
                        for t in tracks:
                            track_name = t.get("name", "").lower()
                            if not any(keyword in track_name for keyword in ['instrumental', 'karaoke', 'backing track', 'minus one']):
                                track = t
                                break
                        
                        if track:
                            break
            
            if not track:
                return jsonify({"error": "Song not found on Spotify"}), 404
                
            track_uri = track.get("uri")
            if not track_uri:
                return jsonify({"error": "Invalid track data"}), 500
                
            # Get additional track details including album artwork
            album = track.get("album", {})
            images = album.get("images", [])
            
            # Play the track
            play_resp = _make_spotify_request(
                "PUT",
                "/me/player/play",
                headers={"Content-Type": "application/json"},
                json={"uris": [track_uri]}
            )
            
            if play_resp.status_code not in (200, 202, 204):
                # If play fails, return track info anyway so UI can show what would have played
                return jsonify({
                    "error": "Failed to play track. Make sure Spotify is open on a device.",
                    "track": {
                        "name": track.get("name"),
                        "artist": ", ".join([a.get("name") for a in track.get("artists", [])]),
                        "uri": track_uri,
                        "duration_ms": track.get("duration_ms"),
                        "album": album.get("name"),
                        "album_art": images[0].get("url") if images else None,
                        "preview_url": track.get("preview_url"),
                        "external_urls": track.get("external_urls", {})
                    }
                }), 400
                
            logger.info(f"Successfully started playing: {track.get('name')} by {track.get('artists', [{}])[0].get('name', 'Unknown')}")
            
            return jsonify({
                "success": True,
                "track": {
                    "name": track.get("name"),
                    "artist": ", ".join([a.get("name") for a in track.get("artists", [])]),
                    "uri": track_uri,
                    "duration_ms": track.get("duration_ms"),
                    "album": album.get("name"),
                    "album_art": images[0].get("url") if images else None,
                    "album_art_small": images[-1].get("url") if images else None,
                    "preview_url": track.get("preview_url"),
                    "external_urls": track.get("external_urls", {}),
                    "popularity": track.get("popularity"),
                    "explicit": track.get("explicit", False)
                }
            })
            
        except SpotifyError:
            return jsonify({"error": "unauthorized"}), 401
        except Exception as e:
            logger.error(f"Error in search and play: {e}")
            return jsonify({"error": "Failed to search and play track"}), 500

    @app.route("/api/spotify/transfer", methods=["POST"])
    def spotify_transfer():
        """Transfer playback to a specific device"""
        try:
            body = request.get_json(silent=True) or {}
            device_id = body.get("device_id", "").strip()
            
            if not device_id:
                return jsonify({"error": "device_id is required"}), 400
                
            auto_play = bool(body.get("play", True))
            
            resp = _make_spotify_request(
                "PUT",
                "/me/player",
                headers={"Content-Type": "application/json"},
                json={"device_ids": [device_id], "play": auto_play}
            )
            
            if resp.status_code not in (200, 202, 204):
                return jsonify({"error": "Failed to transfer playback"}), resp.status_code
                
            return jsonify({"success": True})
            
        except SpotifyError:
            return jsonify({"error": "unauthorized"}), 401
        except Exception as e:
            logger.error(f"Error transferring playback: {e}")
            return jsonify({"error": "Failed to transfer playback"}), 500

    @app.route("/api/spotify/play", methods=["POST"])
    def spotify_play():
        """Start/resume playback"""
        try:
            body = request.get_json(silent=True) or {}
            payload = {}
            
            if body.get("uris"):
                payload["uris"] = body["uris"]
            if isinstance(body.get("position_ms"), int):
                payload["position_ms"] = body["position_ms"]
                
            resp = _make_spotify_request(
                "PUT",
                "/me/player/play",
                headers={"Content-Type": "application/json"},
                json=payload if payload else None
            )
            
            if resp.status_code not in (200, 202, 204):
                return jsonify({"error": "Failed to start playback"}), resp.status_code
                
            return jsonify({"success": True})
            
        except SpotifyError:
            return jsonify({"error": "unauthorized"}), 401
        except Exception as e:
            logger.error(f"Error starting playback: {e}")
            return jsonify({"error": "Failed to start playback"}), 500

    @app.route("/api/spotify/pause", methods=["POST"])
    def spotify_pause():
        """Pause playback"""
        try:
            resp = _make_spotify_request("PUT", "/me/player/pause")
            
            if resp.status_code not in (200, 202, 204):
                return jsonify({"error": "Failed to pause playback"}), resp.status_code
                
            return jsonify({"success": True})
            
        except SpotifyError:
            return jsonify({"error": "unauthorized"}), 401
        except Exception as e:
            logger.error(f"Error pausing playback: {e}")
            return jsonify({"error": "Failed to pause playback"}), 500

    @app.route("/api/spotify/repeat", methods=["POST"]) 
    def spotify_repeat():
        """Set repeat mode (off, track, context). Defaults to off."""
        try:
            body = request.get_json(silent=True) or {}
            state = (body.get("state") or "off").lower()
            if state not in ("off", "track", "context"):
                return jsonify({"error": "invalid_state"}), 400
            resp = _make_spotify_request("PUT", "/me/player/repeat", params={"state": state})
            if resp.status_code not in (200, 202, 204):
                return jsonify({"error": "Failed to set repeat"}), resp.status_code
            return jsonify({"success": True})
        except SpotifyError:
            return jsonify({"error": "unauthorized"}), 401
        except Exception as e:
            logger.error(f"Error setting repeat: {e}")
            return jsonify({"error": "Failed to set repeat"}), 500

    @app.route("/api/spotify/shuffle", methods=["POST"]) 
    def spotify_shuffle():
        """Set shuffle state (true/false). Defaults to false."""
        try:
            body = request.get_json(silent=True) or {}
            state = bool(body.get("state", False))
            resp = _make_spotify_request("PUT", "/me/player/shuffle", params={"state": str(state).lower()})
            if resp.status_code not in (200, 202, 204):
                return jsonify({"error": "Failed to set shuffle"}), resp.status_code
            return jsonify({"success": True})
        except SpotifyError:
            return jsonify({"error": "unauthorized"}), 401
        except Exception as e:
            logger.error(f"Error setting shuffle: {e}")
            return jsonify({"error": "Failed to set shuffle"}), 500

    @app.route("/api/spotify/seek", methods=["POST"])
    def spotify_seek():
        """Seek to position in track"""
        try:
            body = request.get_json(silent=True) or {}
            position_ms = body.get("position_ms")
            
            if not isinstance(position_ms, int) or position_ms < 0:
                return jsonify({"error": "position_ms must be a non-negative integer"}), 400
                
            resp = _make_spotify_request(
                "PUT",
                "/me/player/seek",
                params={"position_ms": position_ms}
            )
            
            if resp.status_code not in (200, 202, 204):
                return jsonify({"error": "Failed to seek"}), resp.status_code
                
            return jsonify({"success": True})
            
        except SpotifyError:
            return jsonify({"error": "unauthorized"}), 401
        except Exception as e:
            logger.error(f"Error seeking: {e}")
            return jsonify({"error": "Failed to seek"}), 500

    @app.route("/api/spotify/search")
    def spotify_search():
        """Search for tracks on Spotify"""
        try:
            query = request.args.get("q", "").strip()
            limit = min(int(request.args.get("limit", 5)), 50)  # Cap at 50
            
            if not query:
                return jsonify({"tracks": []})
                
            resp = _make_spotify_request(
                "GET",
                "/search",
                params={"q": query, "type": "track", "limit": limit}
            )
            
            if resp.status_code != 200:
                return jsonify({"error": "Search failed"}), resp.status_code
                
            data = resp.json()
            tracks = data.get("tracks", {}).get("items", [])
            
            results = []
            for track in tracks[:limit]:
                results.append({
                    "uri": track.get("uri"),
                    "name": track.get("name"),
                    "artists": ", ".join([a.get("name") for a in track.get("artists", [])]),
                    "duration_ms": track.get("duration_ms"),
                    "album": track.get("album", {}).get("name"),
                })
            return jsonify({"tracks": results})
        except SpotifyError:
            return jsonify({"error": "unauthorized"}), 401
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return jsonify({"error": "Search failed"}), 500

    @app.route("/api/spotify/saved-tracks")
    def spotify_saved_tracks():
        """List user's saved tracks, optionally filtered by language via ?lang=fr.

        Requires scope: user-library-read
        """
        try:
            lang_filter = (request.args.get("lang") or "").strip().lower() or None
            limit = int(request.args.get("limit", 50))
            limit = max(1, min(limit, 200))

            # paginate /me/tracks
            items: List[Dict[str, Any]] = []
            fetched = 0
            offset = 0
            while fetched < limit:
                page_size = min(50, limit - fetched)
                resp = _make_spotify_request("GET", f"/me/tracks?limit={page_size}&offset={offset}")
                if resp.status_code != 200:
                    return jsonify({"error": "spotify_api_error"}), 502
                payload = resp.json()
                page_items = payload.get("items", [])
                if not page_items:
                    break
                items.extend(page_items)
                fetched += len(page_items)
                offset += len(page_items)

            results: List[Dict[str, Any]] = []
            for it in items:
                tr = it.get("track") or {}
                name = tr.get("name") or ""
                artists = ", ".join([a.get("name") for a in (tr.get("artists") or []) if a.get("name")])
                if lang_filter:
                    code = detect_language_for_track(name, artists)
                    if not code:
                        continue
                    # Normalize some cases: 'pt-PT' -> 'pt'
                    code = code.split("-")[0]
                    if code != lang_filter:
                        continue
                results.append({
                    "uri": tr.get("uri"),
                    "name": name,
                    "artists": artists,
                    "album": (tr.get("album") or {}).get("name"),
                    "images": (tr.get("album") or {}).get("images", []),
                    "duration_ms": tr.get("duration_ms"),
                })

            return jsonify({"items": results})
        except SpotifyError as e:
            return jsonify({"error": str(e)}), 401
        except Exception as e:
            logger.error(f"Error fetching saved tracks: {e}")
            return jsonify({"error": "server_error"}), 500
            

    @app.route("/api/spotify/devices")
    def spotify_devices():
        """Get available Spotify devices"""
        try:
            resp = _make_spotify_request("GET", "/me/player/devices")
            
            if resp.status_code != 200:
                return jsonify({"error": "Failed to get devices"}), resp.status_code
                
            return jsonify(resp.json())
            
        except SpotifyError:
            return jsonify({"error": "unauthorized"}), 401
        except Exception as e:
            logger.error(f"Error getting devices: {e}")
            return jsonify({"error": "Failed to get devices"}), 500

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error"}), 500

    return app


if __name__ == "__main__":
    try:
        app = create_app()
        
        # Configuration
        use_https = _get_env("USE_HTTPS", "false").lower() in ("1", "true", "yes")
        ssl_context = "adhoc" if use_https else None
        host = "127.0.0.1" if not use_https else "localhost"
        port = int(_get_env("PORT", "5000"))
        debug = _get_env("FLASK_ENV", "production") == "development"
        
        logger.info(f"Starting server on {'https' if use_https else 'http'}://{host}:{port}")
        logger.info(f"Debug mode: {debug}")
        logger.info(f"Spotify redirect URI: {_get_env('SPOTIFY_REDIRECT_URI', 'http://localhost:5000/callback/spotify')}")
        
        app.run(
            debug=debug,
            host=host,
            port=port,
            ssl_context=ssl_context
        )
        
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        exit(1)
