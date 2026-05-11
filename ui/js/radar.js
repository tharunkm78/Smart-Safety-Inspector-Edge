/**
 * radar.js
 * Canvas-based radar with rotating sweep and severity-coded blips.
 */

class Radar {
  constructor(canvasEl) {
    this.canvas = canvasEl;
    this.ctx    = canvasEl.getContext('2d');
    this.W      = canvasEl.width;
    this.H      = canvasEl.height;
    this.cx     = this.W / 2;
    this.cy     = this.H / 2;
    this.radius = Math.min(this.W, this.H) / 2 - 6;
    this.angle  = 0;           // current sweep angle (radians)
    this.speed  = 0.025;       // radians per frame
    this.blips  = [];          // active blips
    this.trails = [];          // fading trail positions

    // Color tokens kept in sync with CSS vars (can't read CSS vars on canvas)
    this.colors = {
      bg:       '#000000',
      grid:     '#0e2010',
      gridLine: '#143318',
      sweep:    'rgba(40, 180, 60, 0.18)',
      sweepEdge:'rgba(58, 200, 80, 0.70)',
      ring:     '#143a18',
      crosshair:'#0d2812',
      label:    '#2a5030',
      critical: '#d92b20',
      medium:   '#d4a017',
      low:      '#3a9c52',
    };

    this._raf = requestAnimationFrame(this._loop.bind(this));
  }

  /** Add a detection blip at a random position within the radar circle. */
  addBlip(priority) {
    const maxR = this.radius * 0.85;
    // Random angle and distance from center
    const a = Math.random() * Math.PI * 2;
    const r = 20 + Math.random() * (maxR - 20);
    const x = this.cx + Math.cos(a) * r;
    const y = this.cy + Math.sin(a) * r;
    const color = priority === 'CRITICAL' ? this.colors.critical
                : priority === 'MEDIUM'   ? this.colors.medium
                : this.colors.low;
    this.blips.push({ x, y, color, born: Date.now(), ttl: 7000, r: 4 });
  }

  clearBlips() { this.blips = []; }

  _drawBackground() {
    const ctx = this.ctx;
    ctx.fillStyle = this.colors.bg;
    ctx.fillRect(0, 0, this.W, this.H);
  }

  _drawGrid() {
    const ctx = this.ctx;
    const rings = 4;
    for (let i = 1; i <= rings; i++) {
      const r = (this.radius / rings) * i;
      ctx.beginPath();
      ctx.arc(this.cx, this.cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = this.colors.gridLine;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    // Crosshair lines
    ctx.strokeStyle = this.colors.crosshair;
    ctx.lineWidth = 1;
    // H line
    ctx.beginPath();
    ctx.moveTo(this.cx - this.radius, this.cy);
    ctx.lineTo(this.cx + this.radius, this.cy);
    ctx.stroke();
    // V line
    ctx.beginPath();
    ctx.moveTo(this.cx, this.cy - this.radius);
    ctx.lineTo(this.cx, this.cy + this.radius);
    ctx.stroke();
    // Diagonal lines (45°)
    const d = this.radius * Math.cos(Math.PI / 4);
    [[-1,-1],[1,-1],[1,1],[-1,1]].forEach(([sx,sy]) => {
      ctx.beginPath();
      ctx.moveTo(this.cx, this.cy);
      ctx.lineTo(this.cx + sx * d, this.cy + sy * d);
      ctx.strokeStyle = this.colors.gridLine;
      ctx.stroke();
    });

    // Outer clip circle border
    ctx.beginPath();
    ctx.arc(this.cx, this.cy, this.radius, 0, Math.PI * 2);
    ctx.strokeStyle = '#1e3820';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  _drawSweep() {
    const ctx = this.ctx;
    const sweepWidth = Math.PI / 2; // 90° trailing glow

    // Save and clip to circle
    ctx.save();
    ctx.beginPath();
    ctx.arc(this.cx, this.cy, this.radius, 0, Math.PI * 2);
    ctx.clip();

    // Sweep gradient (trailing arc)
    const grad = ctx.createConicalGradient
      ? null // fallback below
      : null;

    // Draw sweep as a filled arc sector with gradient fade
    // We simulate this with multiple thin slices
    const slices = 30;
    for (let i = 0; i < slices; i++) {
      const frac    = i / slices;
      const sliceA  = this.angle - sweepWidth * frac;
      const sliceA2 = this.angle - sweepWidth * (frac + 1 / slices);
      const alpha   = (1 - frac) * 0.5;
      ctx.beginPath();
      ctx.moveTo(this.cx, this.cy);
      ctx.arc(this.cx, this.cy, this.radius, sliceA2, sliceA);
      ctx.closePath();
      ctx.fillStyle = `rgba(40, 200, 70, ${alpha})`;
      ctx.fill();
    }

    // Bright leading edge line
    ctx.beginPath();
    ctx.moveTo(this.cx, this.cy);
    ctx.lineTo(
      this.cx + Math.cos(this.angle) * this.radius,
      this.cy + Math.sin(this.angle) * this.radius,
    );
    ctx.strokeStyle = this.colors.sweepEdge;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.restore();
  }

  _drawBlips() {
    const ctx  = this.ctx;
    const now  = Date.now();
    this.blips = this.blips.filter(b => now - b.born < b.ttl);
    this.blips.forEach(b => {
      const age  = now - b.born;
      const life = 1 - age / b.ttl;
      // Outer pulse ring
      const pulseR = b.r + (1 - life) * 8;
      ctx.beginPath();
      ctx.arc(b.x, b.y, pulseR, 0, Math.PI * 2);
      ctx.strokeStyle = b.color.replace(')', `, ${life * 0.5})`).replace('rgb', 'rgba');
      ctx.lineWidth = 1;
      ctx.stroke();
      // Core dot
      ctx.beginPath();
      ctx.arc(b.x, b.y, b.r * life, 0, Math.PI * 2);
      ctx.fillStyle = b.color;
      ctx.globalAlpha = 0.8 + life * 0.2;
      ctx.fill();
      ctx.globalAlpha = 1;
    });
  }

  _drawCenter() {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(this.cx, this.cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#3afc5a';
    ctx.fill();
  }

  _drawLabels() {
    const ctx = this.ctx;
    ctx.font = '8px "Share Tech Mono", monospace';
    ctx.fillStyle = this.colors.label;
    ctx.textAlign = 'center';
    ctx.fillText('N', this.cx, this.cy - this.radius + 10);
    ctx.fillText('S', this.cx, this.cy + this.radius - 3);
    ctx.textAlign = 'left';
    ctx.fillText('E', this.cx + this.radius - 12, this.cy + 3);
    ctx.textAlign = 'right';
    ctx.fillText('W', this.cx - this.radius + 12, this.cy + 3);
    ctx.textAlign = 'left';
  }

  _loop() {
    this.angle = (this.angle + this.speed) % (Math.PI * 2);
    this._drawBackground();
    this._drawGrid();
    this._drawSweep();
    this._drawBlips();
    this._drawCenter();
    this._drawLabels();
    this._raf = requestAnimationFrame(this._loop.bind(this));
  }

  destroy() { cancelAnimationFrame(this._raf); }
}
