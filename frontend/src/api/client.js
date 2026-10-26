/**
 * Thin API client for the StudyGuard backend.
 * Requests go through the Vite dev proxy (`/api` -> backend) in development.
 */

async function getJSON(path) {
  const res = await fetch(path);
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

// Session APIs
export function getActiveStudent() {
  return getJSON('/api/students/active');
}

export function startSession(studentId) {
  return postJSON('/api/sessions/start', {
    student_id: studentId,
  });
}

export function getActiveSession() {
  return getJSON('/api/sessions/active');
}

export function endSession(sessionId) {
  return postJSON(`/api/sessions/${sessionId}/end`, {});
}

// Dashboard APIs
export function getDashboardOverview() {
  return getJSON('/api/dashboard/overview');
}

export function getDashboardTrend(days = 7) {
  return getJSON(`/api/dashboard/trend?days=${days}`);
}

export function getDashboardRecent(limit = 5) {
  return getJSON(`/api/dashboard/recent?limit=${limit}`);
}

// Current live session + latest behavior/focus
export function getLiveSessionStatus() {
  return getJSON('/api/sessions/live');
}

// ── Students ───────────────────────────────────────────────────────────────
export function getStudents() {
  return getJSON('/api/students');
}

export function createStudent(name, email) {
  return postJSON('/api/students', { name, email: email || null });
}

export function setActiveStudent(studentId) {
  return postJSON('/api/students/active', { student_id: studentId });
}

// ── Session history ────────────────────────────────────────────────────────
export function getSessions(studentId, limit = 50) {
  const q = new URLSearchParams({ limit: String(limit) });
  if (studentId != null) q.set('student_id', String(studentId));
  return getJSON(`/api/sessions?${q}`);
}

// ── Analytics ──────────────────────────────────────────────────────────────
export function getSessionStats(sessionId) {
  return getJSON(`/api/analytics/sessions/${sessionId}`);
}

export function getWeeklyAnalytics(days = 7) {
  return getJSON(`/api/analytics/weekly?days=${days}`);
}

// ── Alerts ─────────────────────────────────────────────────────────────────
export function getAlerts({ sessionId, limit = 50 } = {}) {
  const q = new URLSearchParams({ limit: String(limit) });
  if (sessionId != null) q.set('session_id', String(sessionId));
  return getJSON(`/api/alerts?${q}`);
}
