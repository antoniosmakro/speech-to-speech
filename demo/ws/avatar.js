// @ts-check
/**
 * Nyx avatar mode: the orb "opens" into a 3D avatar once a conversation
 * connects, lip-synced to the assistant's voice.
 *
 * Everything avatar-related lives here — the rest of the app only calls the
 * tiny `avatarOrb` surface below, so with the toggle off (or this module
 * unavailable) the demo behaves byte-identically to upstream.
 *
 * How it works
 *  - The renderer is @myned-ai/avatar-chat-widget's renderer-only entry
 *    (`AvatarChat.mountAvatar`), lazy-loaded from a CDN only when the toggle
 *    is on and a conversation starts. It mounts inside the orb's `.circle-core`
 *    and returns a controller (`setChatState`, `updateBlendshapes`,
 *    `updateAudioRMS`, `setBackgroundColor`).
 *  - Lip-sync: the s2s client dispatches each decoded assistant-audio delta
 *    (`assistant-audio`, PCM float32 16 kHz). We forward it to the bundled
 *    server route `/api/wav2arkit`, which returns 52 ARKit blendshape weights
 *    at 30 FPS. Frames are scheduled against the playback clock so the mouth
 *    moves with the audio the user actually hears.
 *  - Playback clock: audio arrives faster than real time, so frame k of a turn
 *    (samples [k, k+1) * 533.3) must wait until playback reaches it. The
 *    playback worklet reports `{queuedMs}` (source-domain ms still queued)
 *    every ~250 ms via the client's `playback-stats` event; played time is
 *    `sentMs - queuedMs`, extrapolated between reports.
 *  - State: the app's `setState()` calls `onAppState()`, which maps app states
 *    to the widget's ChatState, keeps the orb's state classes running, and
 *    repaints the avatar's scene backdrop with the orb's own `--glow` color
 *    mixed into the page dark (the orb core never shows the full-strength
 *    color; ~12 % matches its tinted fill).
 */

/** Pinned renderer library + avatar asset (jsDelivr, versioned). */
const LIB_URL =
  "https://cdn.jsdelivr.net/npm/@myned-ai/avatar-chat-widget@1.15.0/dist/avatar-chat-widget.umd.js";
const ASSET_URL =
  "https://cdn.jsdelivr.net/npm/@myned-ai/avatar-chat-widget@1.15.0/public/asset/nyx.zip";

const SAMPLE_RATE = 16000;
const FPS = 30;
const SAMPLES_PER_FRAME = SAMPLE_RATE / FPS; // 533.33
/** The avatar's 3D scene backdrop: the page's stage-backdrop color where the
 *  orb sits. Constant — all state coloring comes from the .avatar-shade
 *  overlay, which carries the orb core's exact --glow-keyed gradient and
 *  inset shadow, so colors/shadows match the stock orb by construction. */
const SCENE_BASE = "#14161f";
/** Lip-sync timing trim, ms. Positive delays the mouth (frames apply later).
 *  Tunable via localStorage `s2s.avatar.offsetMs` while calibrating. */
const OFFSET_MS_DEFAULT = 40;

/** App state -> widget ChatState. The widget has no "thinking" state; staying
 *  on Listening through `processing` reads naturally (she flips to Responding
 *  the moment audio arrives). */
const CHAT_STATE = {
  listening: "Listening",
  "user-speaking": "Listening",
  processing: "Listening",
  "ai-speaking": "Responding",
};

/** States in which the avatar is shown (the reveal happens on connect). */
const LIVE_STATES = new Set(["listening", "user-speaking", "processing", "ai-speaking"]);

const store = {
  get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
};

function base64FromInt16(int16) {
  let binary = "";
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, /** @type {any} */ (bytes.subarray(i, i + CHUNK)));
  }
  return btoa(binary);
}

class AvatarOrb {
  constructor() {
    this._available = false; // deploy has the lip-sync route
    this._enabled = true;    // user setting (persisted by main.js)
    this._controller = null; // widget MountedAvatar once loaded
    this._libPromise = null;
    this._client = null;     // S2sWsRealtimeClient while attached
    this._listeners = [];
    /** @type {WebSocket | null} lip-sync socket */
    this._ws = null;
    this._wsReady = false;
    this._appState = "idle";
    this._offsetMs = Number(store.get("s2s.avatar.offsetMs", "")) || OFFSET_MS_DEFAULT;

    // Per-turn lip-sync state
    this._turnActive = false;
    this._sentSamples = 0;
    this._frames = [];       // frameIndex -> weights
    this._rmsByFrame = [];   // frameIndex -> rms of its audio window
    this._appliedFrame = -1;
    this._flushed = false;
    // Playback clock model (source-domain ms)
    this._statsQueuedMs = 0;
    this._statsSentMs = 0;
    this._statsAt = 0;
    this._raf = 0;
    // Partial frame accumulator for per-frame RMS
    this._rmsCarry = { sum: 0, n: 0 };
  }

  /** Deploy capability, from /api/config. */
  configure({ available }) {
    this._available = !!available;
  }

  /** User toggle. Off mid-call hides the avatar and stops lip-sync. */
  setEnabled(enabled) {
    this._enabled = !!enabled;
    if (!this._enabled) {
      this._setLive(false);
      this._endTurn();
    } else if (this._client) {
      void this._load();
      this._openWs();
      this.onAppState(this._appState);
    }
  }

  get active() {
    return this._available && this._enabled;
  }

  /** A conversation is starting: begin the lazy load + lip-sync socket now so
   *  the reveal is ready by the time the session connects. */
  attach(client) {
    this.detach();
    this._client = client;
    if (!this.active) return;
    void this._load();
    this._openWs();

    const on = (type, fn) => {
      client.addEventListener(type, fn);
      this._listeners.push([type, fn]);
    };
    on("assistant-audio", (e) => this._onAssistantAudio(e.detail.samples));
    on("playback-stats", (e) => this._onStats(e.detail));
    on("response-finished", () => this._onResponseFinished());
    on("status", (e) => {
      if (e.detail.status === "user-speaking") this._onBargeIn();
    });
  }

  /** Conversation over: back to the stock orb. The renderer stays cached for
   *  an instant reveal on reconnect. */
  detach() {
    for (const [type, fn] of this._listeners) this._client?.removeEventListener(type, fn);
    this._listeners = [];
    this._client = null;
    this._endTurn();
    this._setLive(false);
    if (this._ws) {
      try { this._ws.close(); } catch { /* ignored */ }
      this._ws = null;
      this._wsReady = false;
    }
    this._controller?.setChatState("Idle", "avatar-orb");
  }

  /** Called from the app's setState() — one line in main.js. */
  onAppState(state) {
    this._appState = state;
    if (!this.active) return;
    this._setLive(LIVE_STATES.has(state));
    this._controller?.setChatState(CHAT_STATE[state] ?? "Idle", "avatar-orb");
  }

  // ── Reveal ────────────────────────────────────────────────────────────────

  _setLive(live) {
    document.getElementById("main-circle")?.classList.toggle("avatar-live", !!live);
  }

  async _load() {
    if (this._controller || this._libPromise) return this._libPromise;
    this._libPromise = new Promise((resolve) => {
      const script = document.createElement("script");
      script.src = LIB_URL;
      script.onload = () => {
        try {
          const ns = /** @type {any} */ (window).AvatarChat;
          const api = ns?.AvatarChat ?? ns; // UMD named-exports vs global
          this._controller = api.mountAvatar({
            container: "#avatar-stage",
            avatarUrl: ASSET_URL,
            backgroundColor: SCENE_BASE,
            onError: (err) => console.warn("[avatar] renderer failed:", err),
          });
          this.onAppState(this._appState); // apply current state + tint
        } catch (err) {
          console.warn("[avatar] mount failed:", err);
        }
        resolve();
      };
      script.onerror = () => {
        console.warn("[avatar] failed to load renderer lib:", libUrl);
        resolve();
      };
      document.head.appendChild(script);
    });
    return this._libPromise;
  }

  // ── Lip-sync transport ───────────────────────────────────────────────────

  _openWs() {
    if (this._ws) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/wav2arkit`);
    this._ws = ws;
    ws.addEventListener("message", (e) => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      if (msg.type === "ready") this._wsReady = true;
      else if (msg.type === "frames") this._onFrames(msg.start, msg.weights);
      else if (msg.type === "error") console.warn("[avatar] lipsync:", msg.message);
    });
    ws.addEventListener("close", () => {
      if (this._ws === ws) { this._ws = null; this._wsReady = false; }
    });
    ws.addEventListener("error", () => { /* close handler runs after */ });
  }

  _send(obj) {
    if (this._ws?.readyState === WebSocket.OPEN) this._ws.send(JSON.stringify(obj));
  }

  // ── Turn lifecycle ───────────────────────────────────────────────────────

  _onAssistantAudio(samples) {
    if (!this.active || !samples?.length) return;
    if (!this._turnActive) this._beginTurn();

    // Per-frame RMS (for the widget's audio-gated gaze), aligned to the same
    // 533.3-sample windows the blendshape frames cover.
    let { sum, n } = this._rmsCarry;
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i];
      sum += s * s;
      n++;
      if (n >= SAMPLES_PER_FRAME) {
        this._rmsByFrame.push(Math.sqrt(sum / n));
        sum = 0;
        n = 0;
      }
    }
    this._rmsCarry = { sum, n };

    // Forward to wav2arkit as PCM16 (it was PCM16 on the wire already).
    const int16 = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    this._send({ type: "audio", audio: base64FromInt16(int16) });
    this._sentSamples += samples.length;
  }

  _beginTurn() {
    this._turnActive = true;
    this._flushed = false;
    this._sentSamples = 0;
    this._frames = [];
    this._rmsByFrame = [];
    this._rmsCarry = { sum: 0, n: 0 };
    this._appliedFrame = -1;
    this._statsQueuedMs = 0;
    this._statsSentMs = 0;
    this._statsAt = performance.now();
    this._send({ type: "reset" });
    this._controller?.enableLiveBlendshapes();
    this._startClock();
  }

  _endTurn() {
    this._turnActive = false;
    cancelAnimationFrame(this._raf);
    this._raf = 0;
    this._frames = [];
    this._rmsByFrame = [];
    this._controller?.disableLiveBlendshapes();
  }

  _onBargeIn() {
    // The client already wiped the playback queue; drop pending frames too.
    if (this._turnActive) {
      this._send({ type: "reset" });
      this._endTurn();
    }
  }

  _onResponseFinished() {
    if (this._turnActive) {
      this._send({ type: "flush" });
      this._flushed = true;
    }
  }

  _onFrames(start, weightsList) {
    for (let i = 0; i < weightsList.length; i++) this._frames[start + i] = weightsList[i];
  }

  // ── Playback-synced frame scheduler ──────────────────────────────────────

  _onStats({ queuedMs }) {
    this._statsQueuedMs = queuedMs;
    this._statsSentMs = (this._sentSamples / SAMPLE_RATE) * 1000;
    this._statsAt = performance.now();
  }

  _startClock() {
    cancelAnimationFrame(this._raf);
    const tick = () => {
      if (!this._turnActive) return;
      const sentMs = (this._sentSamples / SAMPLE_RATE) * 1000;
      // Played = sent - queued at the last stats report, advanced in real time
      // since (audio only drains at 1x), clamped to what exists.
      const playedAtStats = Math.max(0, this._statsSentMs - this._statsQueuedMs);
      const playheadMs = Math.min(sentMs, playedAtStats + (performance.now() - this._statsAt));

      const target = Math.floor((playheadMs - this._offsetMs) / (1000 / FPS));
      if (target > this._appliedFrame) {
        // Apply the newest due frame (skipping any we fell behind on).
        for (let k = target; k > this._appliedFrame; k--) {
          const weights = this._frames[k];
          if (weights) {
            this._controller?.updateBlendshapes(weights);
            const rms = this._rmsByFrame[k];
            if (rms !== undefined) this._controller?.updateAudioRMS?.(rms);
            break;
          }
        }
        this._appliedFrame = target;
      }

      // Turn is over once everything sent has played out (plus a beat).
      if (this._flushed && playheadMs >= sentMs - 1) {
        this._endTurn();
        return;
      }
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }
}

export const avatarOrb = new AvatarOrb();
