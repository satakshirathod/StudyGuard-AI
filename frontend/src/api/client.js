/**
 * Thin API client for the StudyGuard backend.
 * Requests go through the Vite dev proxy (`/api` -> backend) in development.
 */

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function postJSON(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data?.detail) detail = data.detail;
    } catch {
      /* keep the HTTP fallback */
    }
    throw new Error(detail);
  }
  return res.json();
}

export function getHealth() {
  return getJSON('/api/health');
}

export function getVideoStatus() {
  return getJSON('/api/video/status');
}

// Start or cancel the gaze calibration run.
export function gazeCalibrate(action) {
  return postJSON('/api/ai/gaze/calibrate', { action });
}

// Start or cancel the screen-distance reference capture.
export function distanceCalibrate(action) {
  return postJSON('/api/ai/distance/calibrate', { action });
}

export const VIDEO_FEED_URL = '/api/video/feed';