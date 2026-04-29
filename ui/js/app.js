/**
 * Smart Safety Inspector — Dashboard Application
 *
 * Connects to the backend via WebSocket (/ws/live) and renders:
 * - Live camera feed with detection overlays
 * - Real-time alert toasts and feed
 * - Detection statistics
 * - Alert history chart
 */

// Try to determine the best WS URL
const getWsUrl = () => {
    const host = window.location.host || 'localhost:8000';
    // If we are on 0.0.0.0 or a weird IP, fallback to localhost
    if (host.startsWith('0.0.0.0')) return `ws://localhost:8000/ws/live`;
    return `ws://${host}/ws/live`;
};

let WS_URL = getWsUrl();
let socket = null;
let frameCounter = 0;
let lastFpsUpdate = 0;
let fps = 0;

// ── DOM Elements ──────────────────────────────────────────────────────────────

const cameraFeed = document.getElementById('cameraFeed');
const overlayCanvas = document.getElementById('overlayCanvas');
const ctx = overlayCanvas.getContext('2d');
const priorityBanner = document.getElementById('priorityBanner');
const priorityText = document.getElementById('priorityText');
const fpsValue = document.getElementById('fpsValue');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const alertList = document.getElementById('alertList');
const toastContainer = document.getElementById('toastContainer');
const criticalCount = document.getElementById('criticalCount');
const highCount = document.getElementById('highCount');
const mediumCount = document.getElementById('mediumCount');
const lowCount = document.getElementById('lowCount');
const modelInfo = document.getElementById('modelInfo');
const connectionInfo = document.getElementById('connectionInfo');
const clearAlertsBtn = document.getElementById('clearAlertsBtn');

// ── State ────────────────────────────────────────────────────────────────────

const alertCounts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
let currentDetections = [];
let alertHistory = [];

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connect() {
    console.log('Attempting WebSocket connection to:', WS_URL);
    statusText.textContent = 'Connecting...';
    
    try {
        socket = new WebSocket(WS_URL);

        socket.onopen = () => {
            console.log('--- WS SUCCESS ---: Connected to backend');
            statusDot.className = 'status-dot connected';
            statusText.textContent = 'Connected';
            connectionInfo.textContent = 'WebSocket: connected';
            loadSystemStatus();
        };

        socket.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleMessage(msg);
        };

        socket.onclose = (e) => {
            console.log('--- WS CLOSED ---: Reason:', e.reason || 'No reason given');
            statusDot.className = 'status-dot';
            statusText.textContent = 'Disconnected — retrying...';
            connectionInfo.textContent = 'WebSocket: disconnected';
            setTimeout(connect, 2000); // Retry faster (2s)
        };

        socket.onerror = (err) => {
            console.error('--- WS ERROR ---:', err);
            statusDot.className = 'status-dot warning';
        };
    } catch (err) {
        console.error('--- WS FATAL ---:', err);
        setTimeout(connect, 5000);
    }
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'detections':
            handleDetections(msg);
            break;
        case 'alert':
            handleAlert(msg.alert);
            break;
        case 'status':
            handleStatus(msg.status);
            break;
    }
}

// ── Detection Handling ────────────────────────────────────────────────────────

function handleDetections(msg) {
    currentDetections = msg.detections || [];
    fps = msg.fps || fps;
    frameCounter++;

    // Update camera feed if image is present
    if (msg.image) {
        cameraFeed.src = 'data:image/jpeg;base64,' + msg.image;
    }

    // FPS display update (every ~500ms)
    const now = performance.now();
    if (now - lastFpsUpdate > 500) {
        fpsValue.textContent = fps.toFixed(1);
        lastFpsUpdate = now;
    }

    // Update priority banner
    const priority = getHighestPriority(currentDetections);
    updatePriorityBanner(priority);

    // Update detection counts
    updateDetectionCounts(currentDetections);

    // Draw overlay on canvas
    drawOverlay(currentDetections);

    // Push to chart
    if (priority !== 'OK' && priority !== 'LOW') {
        pushAlertToChart(priority);
    }
}

function getHighestPriority(detections) {
    if (!detections.length) return 'OK';
    const order = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
    return detections.reduce((best, d) => {
        return order[d.priority] < order[best] ? d.priority : best;
    }, 'LOW');
}

function updatePriorityBanner(priority) {
    priorityBanner.className = 'priority-banner ' + priority;
    priorityText.textContent = priority;
}

function updateDetectionCounts(detections) {
    const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
    for (const d of detections) {
        if (counts[d.priority] !== undefined) counts[d.priority]++;
    }
    criticalCount.textContent = counts.CRITICAL;
    highCount.textContent     = counts.HIGH;
    mediumCount.textContent   = counts.MEDIUM;
    lowCount.textContent      = counts.LOW;
}

// ── Canvas Overlay ────────────────────────────────────────────────────────────

function drawOverlay(detections) {
    if (!cameraFeed.naturalWidth) return;

    // Resize canvas to match display image
    const rect = cameraFeed.getBoundingClientRect();
    overlayCanvas.width  = rect.width;
    overlayCanvas.height = rect.height;
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const scaleX = overlayCanvas.width  / cameraFeed.naturalWidth;
    const scaleY = overlayCanvas.height / cameraFeed.naturalHeight;

    const colors = {
        CRITICAL: '#ff4757',
        HIGH:     '#ff7b00',
        MEDIUM:   '#ffc107',
        LOW:      '#2ed573',
    };

    for (const det of detections) {
        const [x1, y1, x2, y2] = det.bbox;
        const sx1 = x1 * scaleX, sy1 = y1 * scaleY;
        const sx2 = x2 * scaleX, sy2 = y2 * scaleY;
        const color = colors[det.priority] || '#fff';

        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

        // Label background
        const label = `${det.class} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = `600 ${Math.max(11, Math.round(12 * scaleX))}px Inter`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = color + 'cc';
        ctx.fillRect(sx1, sy1 - 20 * scaleY, tw + 8, 18 * scaleY);

        // Label text
        ctx.fillStyle = '#fff';
        ctx.fillText(label, sx1 + 4, sy1 - 6 * scaleY);
    }
}

// ── Alert Handling ────────────────────────────────────────────────────────────

function handleAlert(alert) {
    // Update counts
    alertCounts[alert.priority] = (alertCounts[alert.priority] || 0) + 1;

    // Add to feed
    addAlertToFeed(alert);

    // Toast
    showToast(alert);

    // Update banner
    if (alert.priority === 'CRITICAL' || alert.priority === 'HIGH') {
        updatePriorityBanner(alert.priority);
    }

    // Push to chart
    pushAlertToChart(alert.priority);
}

function addAlertToFeed(alert) {
    // Remove empty state
    const empty = alertList.querySelector('.empty-state');
    if (empty) empty.remove();

    const time = new Date(alert.timestamp || Date.now()).toLocaleTimeString();
    const item = document.createElement('div');
    item.className = `alert-item ${alert.priority}`;
    item.innerHTML = `
        <div class="alert-item-header">
            <span class="alert-class">${escapeHtml(alert.class_name)}</span>
            <span class="alert-time">${time}</span>
        </div>
        <div class="alert-confidence">${(alert.confidence * 100).toFixed(1)}% conf &nbsp;·&nbsp; ${alert.priority}</div>
    `;
    alertList.prepend(item);

    // Limit feed size
    while (alertList.children.length > 50) {
        alertList.lastChild.remove();
    }
}

function showToast(alert) {
    const toast = document.createElement('div');
    toast.className = `toast ${alert.priority}`;
    toast.innerHTML = `
        <div class="toast-title">[${alert.priority}] ${escapeHtml(alert.class_name)}</div>
        <div class="toast-body">${(alert.confidence * 100).toFixed(1)}% confidence detected</div>
    `;
    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

clearAlertsBtn.addEventListener('click', () => {
    alertList.innerHTML = '<div class="empty-state">No alerts detected</div>';
    alertCounts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
    criticalCount.textContent = '0';
    highCount.textContent = '0';
    mediumCount.textContent = '0';
    lowCount.textContent = '0';
});

// ── Status ────────────────────────────────────────────────────────────────────

async function loadSystemStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        modelInfo.textContent = `Model: ${data.model_loaded ? data.model_path.split('/').pop() : 'not loaded'}`;
        if (data.num_classes) {
            modelInfo.textContent += ` · ${data.num_classes} classes`;
        }
    } catch (e) {
        modelInfo.textContent = 'Model: unavailable';
    }
}

function handleStatus(status) {
    // Reserved for system status updates via WS
}

// ── Utility ───────────────────────────────────────────────────────────────────

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Start ─────────────────────────────────────────────────────────────────────

connect();
