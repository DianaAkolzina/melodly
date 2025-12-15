/* Meelange Frontend with Auto-Play & Spotify Visuals */

const els = {
  songSelect: document.getElementById('songSelect'),
  songList: document.getElementById('songList'),
  searchInput: document.getElementById('searchInput'),
  toLang: document.getElementById('toLanguage'),
  fromLang: document.getElementById('fromLanguage'),
  playPause: document.getElementById('playPauseBtn'),
  timeDisplay: document.getElementById('timeDisplay'),
  lyricsList: document.getElementById('lyricsList'),
  translationList: document.getElementById('translationList'),
  grammarBox: document.getElementById('grammarBox'),
  audio: document.getElementById('audio'),
  songMeta: document.getElementById('songMeta'),
  spotifyStatus: document.getElementById('spotifyStatus'),
  spotifyConnectBtn: document.getElementById('spotifyConnectBtn'),
  spotifyLoadSavedBtn: document.getElementById('spotifyLoadSavedBtn'),
  spotifyTrackSelect: document.getElementById('spotifyTrackSelect'),
  spotifyUriInput: document.getElementById('spotifyUriInput'),
  useSpotifyChk: document.getElementById('useSpotifyChk'),
  // New elements for enhanced features
  albumArt: document.getElementById('albumArt'),
  trackTitle: document.getElementById('trackTitle'),
  trackArtist: document.getElementById('trackArtist'),
  loadingOverlay: document.getElementById('loadingOverlay'),
  loadingMessage: document.getElementById('loadingMessage'),
  statusMessage: document.getElementById('statusMessage')
};

let currentSong = null;
let currentSpotifyTrack = null; // Store current Spotify track info
let isPlaying = false;
let useAudio = false; // toggled if audio is loaded
let rafId = null;
let startPerf = 0;
let pausedAt = 0;
let allSongs = [];
let selectedLangFilter = "";

// Spotify state
let hasSpotifyAuth = false;
let useSpotify = false; // controlled by UI checkbox
let spotifyPlayer = null;
let spotifyDeviceId = null;
let lastSpotifyPosMs = 0;
let lastSpotifyTs = 0;
let spotifyDeviceReady = false;
let spotifyCurrent = { name: null, artists: null };

// Enhanced UI functions
function showLoading(message = 'Loading...') {
  if (els.loadingOverlay && els.loadingMessage) {
    els.loadingMessage.textContent = message;
    els.loadingOverlay.style.display = 'flex';
  }
}

function hideLoading() {
  if (els.loadingOverlay) {
    els.loadingOverlay.style.display = 'none';
  }
}

function showStatus(message, type = 'info', duration = 5000) {
  if (els.statusMessage) {
    els.statusMessage.textContent = message;
    els.statusMessage.className = `status-message ${type} show`;
    
    setTimeout(() => {
      els.statusMessage.classList.remove('show');
    }, duration);
  } else {
    // Fallback to console if status element doesn't exist
    console.log(`[${type.toUpperCase()}] ${message}`);
  }
}

function updateTrackVisuals(track) {
  // Update album art if element exists
  if (els.albumArt && track.album_art) {
    els.albumArt.src = track.album_art;
    els.albumArt.style.display = 'block';
  }

  // Update track info if elements exist
  if (els.trackTitle) {
    els.trackTitle.textContent = track.name || 'Unknown Track';
  }
  
  if (els.trackArtist) {
    els.trackArtist.textContent = track.artist || '';
  }

  // Update song meta with additional info
  if (els.songMeta) {
    const metaParts = [];
    if (currentSong?.level) metaParts.push(`Level: ${currentSong.level}`);
    if (currentSong?.from_language && currentSong?.to_language) {
      metaParts.push(`${currentSong.from_language.toUpperCase()}→${currentSong.to_language.toUpperCase()}`);
    }
    if (track.album) metaParts.push(`Album: ${track.album}`);
    if (track.duration_ms) {
      const duration = Math.floor(track.duration_ms / 1000);
      metaParts.push(`Duration: ${fmtTime(duration)}`);
    }
    els.songMeta.textContent = metaParts.filter(Boolean).join(' • ');
  }
}

function fmtTime(sec) {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60).toString().padStart(2, '0');
  const ss = (s % 60).toString().padStart(2, '0');
  return `${m}:${ss}`;
}

function findActiveIndex(time, lines) {
  let idx = 0;
  for (let i = 0; i < lines.length; i++) {
    if (time >= lines[i].time) idx = i;
    else break;
  }
  return idx;
}

function renderSong(song) {
  els.lyricsList.innerHTML = '';
  els.translationList.innerHTML = '';
  els.grammarBox.textContent = 'Select a line to see notes.';

  if (!song.lines || song.lines.length === 0) {
    els.lyricsList.innerHTML = '<li>No lyrics available</li>';
    els.translationList.innerHTML = '<li>No translations available</li>';
    return;
  }

  song.lines.forEach((line, i) => {
    const li = document.createElement('li');
    li.dataset.index = i;
    li.innerHTML = `<span class="time">${fmtTime(line.time)}</span><span class="text">${line.text}</span>`;
    li.addEventListener('click', () => {
      setCurrentTime(line.time);
      showGrammar(i);
    });
    els.lyricsList.appendChild(li);

    const liT = document.createElement('li');
    liT.dataset.index = i;
    liT.innerHTML = `<span class="text">${line.translation}</span>`;
    liT.addEventListener('click', () => {
      setCurrentTime(line.time);
      showGrammar(i);
    });
    els.translationList.appendChild(liT);
  });
}

function showGrammar(index) {
  const line = currentSong?.lines?.[index];
  if (!line) return;
  els.grammarBox.innerHTML = line.grammar || 'No notes for this line.';
}

function highlight(index) {
  document.querySelectorAll('#lyricsList li').forEach(li => li.classList.remove('active'));
  document.querySelectorAll('#translationList li').forEach(li => li.classList.remove('active'));
  const l = document.querySelector(`#lyricsList li[data-index="${index}"]`);
  const r = document.querySelector(`#translationList li[data-index="${index}"]`);
  if (l) l.classList.add('active');
  if (r) r.classList.add('active');
}

function setCurrentTime(t) {
  if (useSpotify && currentSpotifyTrack) {
    // Seek Spotify to requested ms
    const ms = Math.floor(t * 1000);
    fetch('/api/spotify/seek', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ position_ms: ms }) });
    updateTimeDisplay(t);
  } else if (useAudio) {
    els.audio.currentTime = Math.min(Math.max(0, t), currentSong.duration || 0);
    updateTimeDisplay(els.audio.currentTime);
  } else {
    pausedAt = Math.min(Math.max(0, t), currentSong.duration || 0);
    startPerf = performance.now() - pausedAt * 1000;
    updateTimeDisplay(pausedAt);
  }
  
  if (currentSong && currentSong.lines) {
    const idx = findActiveIndex(t, currentSong.lines);
    highlight(idx);
  }
}

function updateTimeDisplay(time) {
  els.timeDisplay.textContent = fmtTime(time);
}

function tick() {
  if (!currentSong) return;
  let cur = 0;
  
  if (useSpotify && currentSpotifyTrack) {
    // Estimate position from last known Spotify state
    if (isPlaying) {
      const dt = performance.now() - lastSpotifyTs;
      cur = (lastSpotifyPosMs + dt) / 1000;
    } else {
      cur = lastSpotifyPosMs / 1000;
    }
  } else if (useAudio) {
    cur = els.audio.currentTime;
  } else {
    cur = (performance.now() - startPerf) / 1000;
    if (cur >= (currentSong.duration || 0)) {
      cur = currentSong.duration || 0;
      pause();
    }
  }
  
  updateTimeDisplay(cur);
  
  if (currentSong.lines) {
    const idx = findActiveIndex(cur, currentSong.lines);
    highlight(idx);
  }
  
  if (isPlaying) rafId = requestAnimationFrame(tick);
}

async function play() {
  if (!currentSong) return;
  
  isPlaying = true;
  els.playPause.textContent = 'Pause';
  
  if (useSpotify && currentSpotifyTrack) {
    try {
      const uri = getSelectedSpotifyURI() || currentSpotifyTrack.uri;
      if (uri) {
        const ms = Math.floor(getCurrentTimeEstimate() * 1000);
        await fetch('/api/spotify/play', { 
          method: 'POST', 
          headers: {'Content-Type': 'application/json'}, 
          body: JSON.stringify({ uris: [uri], position_ms: ms }) 
        });
      } else {
        // If no URI is selected, just resume current playback on device
        await fetch('/api/spotify/play', { method: 'POST' });
      }
    } catch (error) {
      console.error('Error playing Spotify track:', error);
      showStatus('Failed to play on Spotify', 'error');
    }
  } else if (useAudio) {
    try {
      await els.audio.play();
    } catch (error) {
      console.error('Error playing audio:', error);
    }
  } else {
    startPerf = performance.now() - pausedAt * 1000;
  }
  
  rafId = requestAnimationFrame(tick);
}

async function pause() {
  isPlaying = false;
  els.playPause.textContent = 'Play';
  
  if (useAudio) {
    els.audio.pause();
  } else {
    pausedAt = (performance.now() - startPerf) / 1000;
  }
  
  if (useSpotify && currentSpotifyTrack) {
    try {
      await fetch('/api/spotify/pause', { method: 'POST' });
    } catch (error) {
      console.error('Error pausing Spotify:', error);
    }
  }
  
  if (rafId) cancelAnimationFrame(rafId);
}

function togglePlay() {
  if (!currentSong) return;
  if (isPlaying) pause(); else play();
}

async function loadSongs() {
  try {
    showLoading('Loading available songs...');
    
    const res = await fetch('/api/songs');
    const data = await res.json();
    allSongs = data.songs || [];

    // Dropdown (legacy layout, if present)
    if (els.songSelect) {
      els.songSelect.innerHTML = '<option value="">Select a song...</option>';
      allSongs.forEach((s) => {
        const opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = `${s.title} — ${s.artist}`;
        els.songSelect.appendChild(opt);
      });
      els.songSelect.addEventListener('change', () => {
        if (els.songSelect.value) {
          selectSong(els.songSelect.value);
        }
      });
    }

    // Sidebar list (current layout)
    renderSongList();
    if (els.searchInput) {
      els.searchInput.addEventListener('input', () => {
        renderSongList(els.searchInput.value);
      });
    }
    // Language chips
    const chips = Array.from(document.querySelectorAll('[data-lang-chip]'));
  if (chips.length) {
    chips.forEach((chip) => {
      chip.addEventListener('click', () => {
        chips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        selectedLangFilter = (chip.getAttribute('data-lang-chip') || '').toLowerCase();
        if (els.fromLang) {
          els.fromLang.value = selectedLangFilter || '';
        }
        renderSongList(els.searchInput?.value || '');
      });
    });
  }
  if (els.fromLang) {
    els.fromLang.addEventListener('change', () => {
      selectedLangFilter = (els.fromLang.value || '').toLowerCase();
      // sync chips UI
      const chips = Array.from(document.querySelectorAll('[data-lang-chip]'));
      chips.forEach(c => c.classList.toggle('active', (c.getAttribute('data-lang-chip') || '').toLowerCase() === selectedLangFilter));
      if (!selectedLangFilter) {
        const allChip = chips.find(c => (c.getAttribute('data-lang-chip') || '') === '');
        if (allChip) allChip.classList.add('active');
      }
      renderSongList(els.searchInput?.value || '');
    });
  }
    
    hideLoading();
    
  } catch (error) {
    console.error('Error loading songs:', error);
    showStatus('Failed to load songs', 'error');
    hideLoading();
  }
}

function renderSongList(query = '') {
  if (!els.songList) return;
  const q = (query || '').toLowerCase();
  els.songList.innerHTML = '';
  const filtered = allSongs.filter((s) => {
    const hay = `${s.title} ${s.artist} ${s.from_language || ''} ${s.level || ''}`.toLowerCase();
    const langMatch = !selectedLangFilter || (s.from_language || '').toLowerCase() === selectedLangFilter;
    return hay.includes(q) && langMatch;
  });
  filtered.forEach((s) => {
    const div = document.createElement('div');
    div.className = 'song-item';
    div.dataset.id = s.id;
    div.innerHTML = `
      <div class="song-title">${s.title || ''}</div>
      <div class="song-artist">${s.artist || ''}</div>
      <div class="song-meta">${(s.from_language || '').toUpperCase()} → ${(s.to_language || '').toUpperCase()}${s.level ? ' • ' + s.level : ''}</div>
    `;
    div.addEventListener('click', () => {
      document.querySelectorAll('.song-item.active').forEach((el) => el.classList.remove('active'));
      div.classList.add('active');
      selectSong(s.id);
    });
    els.songList.appendChild(div);
  });
}

async function selectSong(id) {
  try {
    showLoading('Loading song data...');
    
    // Load song data
    const res = await fetch(`/api/song/${id}`);
    const song = await res.json();
    currentSong = song;
    
    renderSong(song);
    els.playPause.disabled = false;
    pausedAt = 0;
    setCurrentTime(0);

    // Reset audio and Spotify state
    useAudio = false;
    els.audio.pause();
    els.audio.removeAttribute('src');
    currentSpotifyTrack = null;
    
    // Hide album art initially
    if (els.albumArt) {
      els.albumArt.style.display = 'none';
    }

    // Update basic song meta (Level + language pair)
    const lvl = song.level ? `Level: ${song.level}` : '';
    const langPair = (song.from_language && song.to_language) ? `${song.from_language.toUpperCase()}→${song.to_language.toUpperCase()}` : '';
    const pieces = [lvl, langPair].filter(Boolean).join(' • ');
    els.songMeta.textContent = pieces;
    
    // Update track display elements if they exist
    if (els.trackTitle) els.trackTitle.textContent = song.title || 'Unknown Song';
    if (els.trackArtist) els.trackArtist.textContent = song.artist || '';

    hideLoading();

    // Auto-find and play Spotify track
    if (hasSpotifyAuth && useSpotify && song.title && song.artist) {
      await autoSearchAndPlaySpotify(song);
    } else {
      // Just search without playing if not using Spotify
      if (hasSpotifyAuth) {
        autoSearchSpotifyForSong(song);
      }
      showStatus(`Loaded: ${song.title} by ${song.artist}`, 'info');
    }
    
  } catch (error) {
    console.error('Error selecting song:', error);
    showStatus('Failed to load song', 'error');
    hideLoading();
  }
}

// Enhanced function that searches AND plays automatically
async function autoSearchAndPlaySpotify(song) {
  try {
    showLoading('Searching Spotify and starting playback...');
    
    const response = await fetch('/api/spotify/search-and-play', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        title: song.title,
        artist: song.artist
      })
    });
    
    const result = await response.json();
    
    if (response.ok && result.success) {
      currentSpotifyTrack = result.track;
      updateTrackVisuals(result.track);
      
      // Update the track select dropdown
      if (els.spotifyTrackSelect) {
        els.spotifyTrackSelect.innerHTML = '';
        const opt = document.createElement('option');
        opt.value = result.track.uri;
        opt.textContent = `${result.track.name} — ${result.track.artist}`;
        opt.selected = true;
        els.spotifyTrackSelect.appendChild(opt);
      }
      
      // Start the tick loop for Spotify playback
      lastSpotifyPosMs = 0;
      lastSpotifyTs = performance.now();
      isPlaying = true;
      els.playPause.textContent = 'Pause';
      rafId = requestAnimationFrame(tick);
      
      showStatus(`🎵 Now playing: "${result.track.name}" by ${result.track.artist}`, 'success');
      
    } else {
      // Even if playback failed, show track info if available
      if (result.track) {
        currentSpotifyTrack = result.track;
        updateTrackVisuals(result.track);
        
        // Update dropdown with found track
        if (els.spotifyTrackSelect && result.track.uri) {
          els.spotifyTrackSelect.innerHTML = '';
          const opt = document.createElement('option');
          opt.value = result.track.uri;
          opt.textContent = `${result.track.name} — ${result.track.artist}`;
          opt.selected = true;
          els.spotifyTrackSelect.appendChild(opt);
        }
      }
      
      if (result.error && result.error.includes('Make sure Spotify is open')) {
        showStatus(
          '❌ ' + result.error + ' Open Spotify on your phone, computer, or web player first.',
          'error', 
          8000
        );
      } else {
        showStatus(`❌ ${result.error || 'Failed to play song'}`, 'error');
      }
    }
    
  } catch (error) {
    console.error('Error with auto search and play:', error);
    showStatus('❌ Network error - please try again', 'error');
  } finally {
    hideLoading();
  }
}

function getSelectedSpotifyURI() {
  const manual = (els.spotifyUriInput?.value || '').trim();
  if (manual) return manual;
  const sel = els.spotifyTrackSelect;
  if (sel && sel.value) return sel.value;
  if (currentSpotifyTrack) return currentSpotifyTrack.uri;
  return null;
}

function getCurrentTimeEstimate() {
  if (useSpotify && currentSpotifyTrack) {
    const dt = performance.now() - lastSpotifyTs;
    return (lastSpotifyPosMs + dt) / 1000;
  }
  if (useAudio) return els.audio.currentTime;
  return (performance.now() - startPerf) / 1000;
}

async function ensureSpotifyAuth() {
  const res = await fetch('/api/spotify/token');
  if (res.status === 200) {
    hasSpotifyAuth = true;
    if (els.spotifyLoadSavedBtn) els.spotifyLoadSavedBtn.disabled = false;
    if (els.spotifyTrackSelect) els.spotifyTrackSelect.disabled = false;
    // Default to Spotify playback when authed
    if (els.useSpotifyChk) {
      els.useSpotifyChk.checked = true;
      useSpotify = true;
    }
  } else {
    hasSpotifyAuth = false;
    if (els.spotifyLoadSavedBtn) els.spotifyLoadSavedBtn.disabled = true;
    if (els.spotifyTrackSelect) els.spotifyTrackSelect.disabled = true;
  }
  return hasSpotifyAuth;
}

function initSpotifySDK() {
  if (!hasSpotifyAuth) return;
  if (spotifyPlayer) return;
  if (!window.Spotify || !window.Spotify.Player) {
    // SDK not loaded yet; try again shortly
    setTimeout(initSpotifySDK, 500);
    return;
  }
  
  spotifyPlayer = new Spotify.Player({
    name: 'Meelange Player',
    getOAuthToken: async cb => {
      try {
        const res = await fetch('/api/spotify/token');
        if (res.status !== 200) return;
        const data = await res.json();
        cb(data.access_token);
      } catch (error) {
        console.error('Error getting Spotify token:', error);
      }
    },
    volume: 0.8,
  });

  spotifyPlayer.addListener('ready', ({ device_id }) => {
    spotifyDeviceId = device_id;
    spotifyDeviceReady = true;
    // Make this device active
    fetch('/api/spotify/transfer', { 
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' }, 
      body: JSON.stringify({ device_id, play: false }) 
    });
    updateSpotifyStatus();
    showStatus('Spotify player ready! 🎉', 'success');
  });
  
  spotifyPlayer.addListener('not_ready', () => { 
    spotifyDeviceReady = false;
    updateSpotifyStatus();
  });
  
  spotifyPlayer.addListener('player_state_changed', state => {
    if (!state) return;
    
    lastSpotifyPosMs = state.position;
    lastSpotifyTs = performance.now();
    
    // Sync play/pause button state
    const paused = state.paused;
    if (paused && isPlaying) {
      // If Spotify paused externally, reflect it
      isPlaying = false;
      els.playPause.textContent = 'Play';
      if (rafId) cancelAnimationFrame(rafId);
    }
    
    const cur = state.track_window?.current_track;
    if (cur) {
      spotifyCurrent.name = cur.name || null;
      spotifyCurrent.artists = (cur.artists || []).map(a => a.name).join(', ');
      updateSpotifyStatus();
    }
  });

  spotifyPlayer.connect();
}

async function loadSavedTracks() {
  const res = await fetch('/api/spotify/saved-tracks');
  if (res.status !== 200) return;
  const data = await res.json();
  const items = data.items || [];
  els.spotifyTrackSelect.innerHTML = '';
  items.forEach(it => {
    const tr = it.track || {};
    const name = `${tr.name || 'Unknown'} — ${(tr.artists && tr.artists.map(a => a.name).join(', ')) || ''}`;
    const opt = document.createElement('option');
    opt.value = tr.uri;
    opt.textContent = name;
    els.spotifyTrackSelect.appendChild(opt);
  });
}

// Enhanced version that searches but doesn't auto-play
async function autoSearchSpotifyForSong(song) {
  const q = `${song.title || ''} ${song.artist || ''}`.trim();
  if (!q) return;
  
  try {
    const res = await fetch(`/api/spotify/search?q=${encodeURIComponent(q)}&limit=5`);
    if (res.status !== 200) return;
    
    const data = await res.json();
    const tracks = data.tracks || [];
    
    if (els.spotifyTrackSelect) {
      els.spotifyTrackSelect.innerHTML = '';
      tracks.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t.uri;
        opt.textContent = `${t.name} — ${t.artists}`;
        els.spotifyTrackSelect.appendChild(opt);
      });
      
      if (tracks[0]) {
        els.spotifyTrackSelect.value = tracks[0].uri;
        // Store the first result as current track for potential visuals
        currentSpotifyTrack = {
          uri: tracks[0].uri,
          name: tracks[0].name,
          artist: tracks[0].artists
        };
      }
    }
    
    updateSpotifyStatus();
  } catch (error) {
    console.error('Error searching Spotify:', error);
  }
}

function updateSpotifyStatus() {
  const parts = [];
  if (!hasSpotifyAuth) {
    els.spotifyStatus.textContent = 'Spotify: Not connected';
    return;
  }
  parts.push('Spotify: Authed');
  if (spotifyDeviceReady) parts.push('Device ready');
  if (spotifyCurrent.name) parts.push(`Track: ${spotifyCurrent.name} — ${spotifyCurrent.artists || ''}`);
  els.spotifyStatus.textContent = parts.join(' • ');
}

async function init() {
  els.playPause.addEventListener('click', togglePlay);
  
  // Load songs
  await loadSongs();

  // Spotify controls
  if (els.spotifyConnectBtn) {
    els.spotifyConnectBtn.addEventListener('click', () => {
      window.location.href = '/auth/spotify/login';
    });
  }
  
  if (els.spotifyLoadSavedBtn) {
    els.spotifyLoadSavedBtn.addEventListener('click', loadSavedTracks);
  }
  
  if (els.useSpotifyChk) {
    els.useSpotifyChk.addEventListener('change', () => {
      useSpotify = els.useSpotifyChk.checked && hasSpotifyAuth;
      if (useSpotify) {
        // Pause local audio if switching to Spotify
        if (useAudio) { 
          els.audio.pause(); 
          useAudio = false; 
        }
      }
    });
  }

  // Initialize Spotify
  const isAuthed = await ensureSpotifyAuth();
  if (isAuthed) {
    initSpotifySDK();
  }
  updateSpotifyStatus();
  
  // Show welcome message
  showStatus('Welcome to Meelange! Connect Spotify to play music, then pick a song.', 'info');
}

// Initialize when page loads
init();
