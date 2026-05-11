/**
 * mock-data.js
 * All simulated configuration and detection scenarios.
 * Swap CAMERAS array with real feed data when integrating the backend.
 */

// ─── Camera Configuration ──────────────────────────────────────
const CAMERAS = [
  { id: 'FEED-01', name: 'ENTRANCE GATE', location: 'ZONE-A' },
];

// ─── Class Metadata ────────────────────────────────────────────
const CLASS_META = {
  fire:          { priority: 'CRITICAL', label: 'FIRE HAZARD' },
  smoke:         { priority: 'CRITICAL', label: 'SMOKE HAZARD' },
  helmet_on:     { priority: 'LOW',      label: 'Hardhat ON' },
  vest_on:       { priority: 'LOW',      label: 'Vest ON' },
  gloves_on:     { priority: 'LOW',      label: 'Gloves ON' },
  boots:         { priority: 'LOW',      label: 'Boots' },
  person:        { priority: 'LOW',      label: 'Person' },
};

// ─── Detection Scenarios ───────────────────────────────────────
const SCENARIO_DURATION = 6000;

const DETECTION_SCENARIOS = [
  {
    label: 'ROUTINE SURVEILLANCE',
    detections: [
      { cls: 'person',    confidence: 0.82, bbox: [300, 80, 440, 360] },
      { cls: 'helmet_on', confidence: 0.88, bbox: [308, 60, 420, 140] },
    ]
  }
];
