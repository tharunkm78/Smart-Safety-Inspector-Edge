/**
 * app.js
 * Main orchestrator: camera feed renderer, bounding boxes,
 * alert system, risk panel, clock, FPS counter, radar blips.
 */

// ─── Constants ─────────────────────────────────────────────────
const PRIORITY_COLOR = {
  CRITICAL: { stroke: '#d92b20', glow: 'rgba(217,43,32,0.8)',  blur: 20, label: '#d92b20' },
  MEDIUM:   { stroke: '#d4a017', glow: 'rgba(212,160,23,0.6)', blur: 14, label: '#d4a017' },
  LOW:      { stroke: '#3a9c52', glow: 'rgba(58,156,82,0.3)',  blur: 6,  label: '#3a9c52' },
};

// ─── Audio ──────────────────────────────────────────────────────
let _audioCtx = null;
function getAudioCtx() {
  if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return _audioCtx;
}
function playBeep(priority) {
  try {
    const ctx  = getAudioCtx();
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);

    if (priority === 'CRITICAL') {
      // Two sharp high-pitched pulses
      osc.frequency.value = 880;
      osc.type = 'square';
      gain.gain.setValueAtTime(0, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0.25, ctx.currentTime + 0.02);
      gain.gain.setValueAtTime(0.25, ctx.currentTime + 0.12);
      gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.14);
      gain.gain.linearRampToValueAtTime(0.25, ctx.currentTime + 0.22);
      gain.gain.setValueAtTime(0.25, ctx.currentTime + 0.30);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.38);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.4);
    } else {
      // Single medium beep
      osc.frequency.value = 520;
      osc.type = 'sine';
      gain.gain.setValueAtTime(0, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.03);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.28);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.3);
    }
  } catch (_) { /* silently ignore if audio blocked */ }
}

// ─── Camera Feed ────────────────────────────────────────────────
class CameraFeed {
  constructor(canvas, config) {
    this.canvas     = canvas;
    this.ctx        = canvas.getContext('2d');
    this.config     = config;
    this.detections = [];
    this.t          = 0;
    this.W          = 0;
    this.H          = 0;

    // Video feed from backend
    this.videoImg = new Image();
    this.videoImg.src = '/api/video_feed';

    // Animated gradient blobs — simulating a dim construction site scene
    this.blobs = [];
    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(canvas.parentElement);
    this._onResize();
    this._raf = requestAnimationFrame(this._loop.bind(this));
  }

  _onResize() {
    const parent = this.canvas.parentElement;
    const dpr    = window.devicePixelRatio || 1;
    const w      = parent.clientWidth;
    const h      = parent.clientHeight;
    this.canvas.width  = w * dpr;
    this.canvas.height = h * dpr;
    this.ctx.scale(dpr, dpr);
    this.W = w;
    this.H = h;
    this.blobs = this._generateBlobs();
  }

  _generateBlobs() {
    const W = this.W || 640;
    const H = this.H || 480;
    const baseColors = [
      [18, 22, 15],  // dark green
      [22, 18, 14],  // dark brown
      [14, 18, 22],  // dark blue-gray
      [20, 20, 18],  // near-black warm
      [16, 22, 20],  // dark teal
    ];
    return Array.from({ length: 7 }, (_, i) => {
      const c = baseColors[i % baseColors.length];
      return {
        x:     Math.random() * W,
        y:     Math.random() * H,
        r:     150 + Math.random() * 200,
        dx:    (Math.random() - 0.5) * 0.25,
        dy:    (Math.random() - 0.5) * 0.25,
        color: c,
      };
    });
  }

  _drawBackground() {
    const ctx = this.ctx;
    const W = this.W, H = this.H;

    // Base fill — almost black with very slight warm tint
    ctx.fillStyle = '#07090a';
    ctx.fillRect(0, 0, W, H);

    // Drift and render blobs
    this.blobs.forEach(b => {
      b.x += b.dx;
      b.y += b.dy;
      if (b.x < -b.r) b.x = W + b.r;
      if (b.x > W + b.r) b.x = -b.r;
      if (b.y < -b.r) b.y = H + b.r;
      if (b.y > H + b.r) b.y = -b.r;

      const grad = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r);
      const [r, g, bl] = b.color;
      grad.addColorStop(0, `rgba(${r},${g},${bl},0.7)`);
      grad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
    });

    // Occasional flicker line
    if (Math.random() < 0.012) {
      const fy = Math.random() * H;
      ctx.fillStyle = 'rgba(255,255,255,0.025)';
      ctx.fillRect(0, fy, W, 2);
    }
  }

  _drawNoise() {
    const ctx = this.ctx;
    const W = this.W, H = this.H;
    for (let i = 0; i < 180; i++) {
      const x = Math.random() * W;
      const y = Math.random() * H;
      const v = Math.floor(Math.random() * 25);
      ctx.fillStyle = `rgba(${v + 8},${v + 12},${v + 8},${(Math.random() * 0.35).toFixed(2)})`;
      ctx.fillRect(x, y, 1, 1);
    }
  }

  _drawScanlines() {
    const ctx = this.ctx;
    const W = this.W, H = this.H;
    ctx.fillStyle = 'rgba(0,0,0,0.13)';
    for (let y = 0; y < H; y += 3) {
      ctx.fillRect(0, y, W, 1);
    }
  }

  _drawDetections() {
    const ctx = this.ctx;
    const scaleX = this.W; // Normalized coordinates, just multiply by canvas width
    const scaleY = this.H; // Normalized coordinates, just multiply by canvas height

    if (this.detections.length > 0 && this.t % 60 === 0) {
      console.log(`[HUD-DEBUG] Drawing ${this.detections.length} detections. Canvas: ${this.W}x${this.H}, Scales: ${scaleX.toFixed(3)},${scaleY.toFixed(3)}`);
    }

    this.detections.forEach(det => {
      const priority = det.priority || 'LOW';
      const labelText = det.class || 'unknown';
      const pal  = PRIORITY_COLOR[priority] || PRIORITY_COLOR.LOW;
      const [rx1, ry1, rx2, ry2] = det.bbox;
      const x  = rx1 * scaleX;
      const y  = ry1 * scaleY;
      const bw = (rx2 - rx1) * scaleX;
      const bh = (ry2 - ry1) * scaleY;

      // ── Anime Aura Glow ────────────────────────────────────
      // Multiple rect strokes, each expanding outward and fading —
      // creates a soft, organic energy bloom rather than a hard glow.
      if (priority !== 'LOW') {
        const auraLayers = [
          { expand: 22, alpha: 0.025 },
          { expand: 16, alpha: 0.055 },
          { expand: 11, alpha: 0.10  },
          { expand: 7,  alpha: 0.18  },
          { expand: 4,  alpha: 0.28  },
          { expand: 2,  alpha: 0.45  },
        ];
        ctx.save();
        auraLayers.forEach(({ expand, alpha }) => {
          ctx.globalAlpha = alpha;
          ctx.strokeStyle = pal.stroke;
          ctx.lineWidth   = 1.5;
          ctx.strokeRect(
            x - expand, y - expand,
            bw + expand * 2, bh + expand * 2,
          );
        });
        ctx.restore();
      }

      // ── Solid edge line ────────────────────────────────────
      ctx.strokeStyle = pal.stroke;
      ctx.lineWidth   = 1.5;
      ctx.globalAlpha = 1;
      ctx.strokeRect(x, y, bw, bh);

      // ── Corner brackets ────────────────────────────────────
      const cs = Math.min(12, bw * 0.15, bh * 0.15);
      ctx.lineWidth = 2;
      ctx.strokeStyle = pal.stroke;
      const drawCorner = (ox, oy, sx, sy) => {
        ctx.beginPath();
        ctx.moveTo(ox, oy + sy * cs);
        ctx.lineTo(ox, oy);
        ctx.lineTo(ox + sx * cs, oy);
        ctx.stroke();
      };
      drawCorner(x,      y,      1,  1);
      drawCorner(x + bw, y,      -1, 1);
      drawCorner(x,      y + bh, 1,  -1);
      drawCorner(x + bw, y + bh, -1, -1);

      // ── Label chip ─────────────────────────────────────────
      const label = `${labelText.toUpperCase()}  ${Math.round(det.confidence * 100)}%`;
      ctx.font = `bold 10px "Share Tech Mono", monospace`;
      const lw = ctx.measureText(label).width + 14;
      const lh = 18;
      const lx = x;
      const ly = y - lh - 2 < 0 ? y + bh + 2 : y - lh - 2;
      ctx.globalAlpha = 1;
      ctx.fillStyle = pal.stroke;
      ctx.fillRect(lx, ly, lw, lh);
      ctx.fillStyle = priority === 'LOW' ? '#000' : '#fff';
      ctx.fillText(label, lx + 7, ly + 12);
    });

    ctx.globalAlpha = 1;
  }

  updateDetections(detections, w, h) {
    this.detections = detections;
    this.origW = w;
    this.origH = h;
  }

  _loop() {
    this.t++;
    if (this.videoImg && this.videoImg.complete && this.videoImg.naturalWidth > 0) {
      this.ctx.drawImage(this.videoImg, 0, 0, this.W, this.H);
    } else {
      this._drawBackground();
      this._drawNoise();
    }
    this._drawDetections();
    this._drawScanlines();
    this._raf = requestAnimationFrame(this._loop.bind(this));
  }

  destroy() {
    cancelAnimationFrame(this._raf);
    this._resizeObserver.disconnect();
  }
}

// ─── Alert Manager ──────────────────────────────────────────────
class AlertManager {
  constructor(container) {
    this.container  = container;
    this.queue      = [];
    this.maxVisible = 3;
    this.totalCount = 0;
  }

  push(scenario, detections) {
    this.totalCount++;
    document.getElementById('totalAlerts').textContent = this.totalCount;

    // Use the backend-provided priority for the alert level
    const priorities = detections.map(d => d.priority || 'LOW');
    const priority = priorities.includes('CRITICAL') ? 'CRITICAL' 
                   : priorities.includes('MEDIUM') ? 'MEDIUM' 
                   : 'LOW';

    if (this.container.children.length >= this.maxVisible) return;

    playBeep(priority);

    const el = document.createElement('div');
    el.className = `mission-alert alert-${priority.toLowerCase()}`;

    const DURATION = 6000;
    el.innerHTML = `
      <div class="alert-inner">
        <div class="alert-header">
          <span class="alert-title">⚠ MISSION ALERT</span>
          <span class="alert-priority-badge ${priority.toLowerCase()}">${priority}</span>
        </div>
        <div class="alert-body">${scenario.label}</div>
        <div class="alert-detail">
          ${detections.length} object(s) detected &nbsp;·&nbsp; ${new Date().toLocaleTimeString()}
        </div>
      </div>
      <div class="alert-timer"><div class="alert-timer-bar" id="atb-${this.totalCount}"></div></div>
    `;

    this.container.appendChild(el);

    // Animate the countdown bar
    const bar = el.querySelector('.alert-timer-bar');
    if (bar) {
      bar.style.transition = `transform ${DURATION}ms linear`;
      requestAnimationFrame(() => {
        requestAnimationFrame(() => { bar.style.transform = 'scaleX(0)'; });
      });
    }

    setTimeout(() => {
      el.classList.add('dismissing');
      setTimeout(() => el.remove(), 350);
    }, DURATION);
  }
}

// ─── Risk Panel ─────────────────────────────────────────────────
class RiskPanel {
  constructor() {
    this.criticalEl = document.getElementById('criticalItems');
    this.mediumEl   = document.getElementById('mediumItems');
    this.lowEl      = document.getElementById('lowItems');
    this.countEl    = document.getElementById('threatCount');
  }

  update(detections) {
    const byPriority = { CRITICAL: [], MEDIUM: [], LOW: [] };
    detections.forEach(d => {
      const priority = d.priority || 'LOW';
      byPriority[priority].push({ ...d, label: d.class });
    });

    this.countEl.textContent = detections.length;
    this._render(this.criticalEl, byPriority.CRITICAL, 'critical');
    this._render(this.mediumEl,   byPriority.MEDIUM,   'medium');
    this._render(this.lowEl,      byPriority.LOW,       'low');
  }

  _render(container, items, cls) {
    if (!items.length) {
      container.innerHTML = '<div class="risk-empty">—</div>';
      return;
    }
    container.innerHTML = items.map(d => `
      <div class="risk-item ${cls}">
        <div class="risk-item-left">
          <div class="risk-item-dot"></div>
          <span class="risk-item-name">${d.label}</span>
        </div>
        <span class="risk-item-conf">${Math.round(d.confidence * 100)}%</span>
      </div>
    `).join('');
  }
}

// ─── Grid Builder ───────────────────────────────────────────────
function buildGrid(cameras) {
  const grid = document.getElementById('feedGrid');
  grid.innerHTML = '';

  // Set grid columns class
  const n = cameras.length;
  const colClass = n <= 1 ? 'cols-1' : n <= 2 ? 'cols-2' : n <= 4 ? 'cols-4' : 'cols-6';
  grid.className = `feed-grid ${colClass}`;

  const feeds = [];

  cameras.forEach(cam => {
    const card = document.createElement('div');
    card.className = 'camera-card';
    card.id = `card-${cam.id}`;

    const canvas = document.createElement('canvas');
    canvas.id = `canvas-${cam.id}`;

    // HUD overlay
    const hud = document.createElement('div');
    hud.className = 'cam-hud';
    hud.innerHTML = `
      <div class="cam-badge">
        <span>${cam.id}</span>
        <span class="rec-dot"></span>
        <span>REC</span>
      </div>
      <div class="cam-timestamp" id="ts-${cam.id}">--:--:--</div>
      <div class="cam-name">${cam.name}</div>
      <div class="cam-location">${cam.location}</div>
      <div class="corner tl"></div>
      <div class="corner tr"></div>
      <div class="corner bl"></div>
      <div class="corner br"></div>
    `;

    card.appendChild(canvas);
    card.appendChild(hud);
    grid.appendChild(card);

    const feed = new CameraFeed(canvas, cam);
    feeds.push({ cam, card, canvas, feed });
  });

  return feeds;
}

// ─── Clock & Uptime ─────────────────────────────────────────────
function startClock(startTime) {
  const clockEl  = document.getElementById('liveClock');
  const dateEl   = document.getElementById('liveDate');
  const uptimeEl = document.getElementById('footerUptime');

  const days = ['SUN','MON','TUE','WED','THU','FRI','SAT'];
  const months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];

  setInterval(() => {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString('en-US', { hour12: false });
    dateEl.textContent  = `${days[now.getDay()]} ${months[now.getMonth()]} ${String(now.getDate()).padStart(2,'0')}, ${now.getFullYear()}`;

    // Uptime
    const sec  = Math.floor((Date.now() - startTime) / 1000);
    const hh   = String(Math.floor(sec / 3600)).padStart(2,'0');
    const mm   = String(Math.floor((sec % 3600) / 60)).padStart(2,'0');
    const ss   = String(sec % 60).padStart(2,'0');
    uptimeEl.textContent = `UPTIME: ${hh}:${mm}:${ss}`;

    // Per-feed timestamps
    CAMERAS.forEach(cam => {
      const ts = document.getElementById(`ts-${cam.id}`);
      if (ts) ts.textContent = now.toLocaleTimeString('en-US', { hour12: false });
    });
  }, 1000);
}

// ─── FPS Counter ────────────────────────────────────────────────
function startFpsCounter() {
  const el = document.getElementById('headerFps');
  let frames = 0, last = performance.now();
  const tick = () => {
    frames++;
    const now = performance.now();
    if (now - last >= 1000) {
      el.textContent = `${frames} FPS`;
      frames = 0;
      last = now;
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ─── Mock Engine ────────────────────────────────────────────────
class MockEngine {
  constructor(feeds, alertMgr, riskPanel, radar) {
    this.feeds      = feeds;
    this.alertMgr   = alertMgr;
    this.riskPanel  = riskPanel;
    this.radar      = radar;
    this.idx        = 0;
    this.detCountEl = document.getElementById('headerDetections');
    this.missionEl  = document.getElementById('footerMission');
    this.radarSigs  = document.getElementById('radarSigs');
    this._tick();
  }

  _tick() {
    const scenario = DETECTION_SCENARIOS[this.idx % DETECTION_SCENARIOS.length];
    this.idx++;

    // Push detections to ALL feeds (mock — same data for all cameras)
    this.feeds.forEach(({ feed, card }) => {
      feed.updateDetections(scenario.detections);

    // Update card border — no full-card glow, just clean border
      card.classList.remove('active-alert-critical', 'active-alert-medium');
    });

    // Risk panel
    this.riskPanel.update(scenario.detections);
    this.detCountEl.textContent = `${scenario.detections.length} DETECTED`;
    this.missionEl.textContent  = `◈ MISSION: ${scenario.label}`;

    // Alert popup only on critical/medium scenarios
    const hasCritical = scenario.detections.some(
      d => (CLASS_META[d.cls] || {}).priority === 'CRITICAL'
    );
    const hasMedium = scenario.detections.some(
      d => (CLASS_META[d.cls] || {}).priority === 'MEDIUM'
    );
    if (hasCritical || hasMedium) {
      this.alertMgr.push(scenario, scenario.detections);
    }

    // Radar blips
    this.radar.clearBlips();
    scenario.detections.forEach(d => {
      const priority = (CLASS_META[d.cls] || { priority: 'LOW' }).priority;
      this.radar.addBlip(priority);
    });
    this.radarSigs.textContent = `${scenario.detections.length} signatures`;

    setTimeout(() => this._tick(), SCENARIO_DURATION);
  }
}

// ─── Bootstrap ──────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  const startTime = Date.now();

  // Build camera grid
  const feeds = buildGrid(CAMERAS);

  // Init subsystems
  const alertMgr  = new AlertManager(document.getElementById('alertOverlay'));
  const riskPanel = new RiskPanel();
  const radar     = new Radar(document.getElementById('radarCanvas'));

  // Connect to actual WebSocket backend
  const ws = new WebSocket(`ws://${window.location.host}/ws/live`);
  let lastThreatHash = "";
  let lastPanelUpdate = 0;
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'detections') {
      const detections = data.detections;
      
      // 1. ALWAYS update camera feeds (must be smooth/real-time for tracking)
      feeds.forEach(({ feed, card }) => {
        feed.updateDetections(detections, data.width, data.height);
        card.classList.remove('active-alert-critical', 'active-alert-medium');
      });
      
      // 2. STABILIZED UI updates for panels
      // We hash based on CLASS and PRIORITY to ignore position/confidence jitter
      const currentHash = detections
        .map(d => `${d.class}:${d.priority}`)
        .sort()
        .join('|');
        
      const now = Date.now();
      const needsUpdate = (currentHash !== lastThreatHash) || (now - lastPanelUpdate > 1000);
      
      if (needsUpdate) {
        riskPanel.update(detections);
        document.getElementById('headerDetections').textContent = `${detections.length} DETECTED`;
        
        // Only trigger Alert popup on meaningful CHANGES to prevent alert-spam
        if (currentHash !== lastThreatHash) {
          const hasCritical = detections.some(d => d.priority === 'CRITICAL');
          const hasMedium = detections.some(d => d.priority === 'MEDIUM');
          if (hasCritical || hasMedium) {
            alertMgr.push({ label: 'THREAT DETECTED' }, detections);
          }
        }
        
        radar.clearBlips();
        detections.forEach(d => radar.addBlip(d.priority));
        document.getElementById('radarSigs').textContent = `${detections.length} signatures`;
        
        lastThreatHash = currentHash;
        lastPanelUpdate = now;
      }

      if (data.fps) {
        document.getElementById('headerFps').textContent = `${Math.round(data.fps)} FPS`;
      }
    }
  };

  // UI helpers
  startClock(startTime);

  // Footer feed status
  document.getElementById('feedStatusLabel').textContent =
    CAMERAS.map(c => `${c.id}: ACTIVE`).join('  |  ');
});
