// Backend stores naive UTC timestamps (no "Z"). Without it the browser treats
// them as local time and shows the wrong hour, so we add it here.
export function parseUtc(value) {
  if (!value) return null;
  const hasZone = /[zZ]|[+-]\d\d:?\d\d$/.test(value);
  const d = new Date(hasZone ? value : `${value}Z`);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function formatDateTime(value) {
  const d = parseUtc(value);
  return d
    ? d.toLocaleString([], { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
    : '—';
}

export function formatDay(value) {
  // value is a plain "YYYY-MM-DD" from the trend API
  if (!value) return '—';
  const d = new Date(`${value}T00:00:00`);
  return Number.isNaN(d.getTime())
    ? value
    : d.toLocaleDateString([], { weekday: 'short', day: 'numeric' });
}

export function formatDuration(seconds) {
  if (seconds == null) return '—';
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function formatScore(score) {
  return score == null ? '—' : Math.round(score);
}

// Badge class for a 0-100 focus score.
export function scoreBadge(score) {
  if (score == null) return 'badge-unknown';
  if (score >= 70) return 'badge-ok';
  if (score >= 40) return 'badge-warn';
  return 'badge-crit';
}
