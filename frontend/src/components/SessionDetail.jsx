import { useEffect, useState } from 'react';
import { getSessionStats, getAlerts } from '../api/client.js';
import { BreakdownBar, DistributionRows } from './Charts.jsx';
import { formatDateTime, formatDuration, formatScore, scoreBadge } from '../utils/format.js';

function Stat({ label, value, hint }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {hint && <span className="stat-hint">{hint}</span>}
    </div>
  );
}

// Full analytics of ONE session: summary, focus/behavior breakdown, gaze, alerts.
export default function SessionDetail({ sessionId }) {
  const [stats, setStats] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    Promise.all([getSessionStats(sessionId), getAlerts({ sessionId, limit: 50 })])
      .then(([s, a]) => {
        if (!alive) return;
        setStats(s);
        setAlerts(a);
      })
      .catch((err) => alive && setError(err.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [sessionId]);

  if (loading) return <div className="loading">Loading session…</div>;
  if (error) return <div className="form-error">Could not load session: {error}</div>;
  if (!stats) return null;

  const s = stats.session;
  const f = stats.focus;

  return (
    <div className="stack">
      <div className="row-between">
        <div>
          <div className="card-title" style={{ margin: 0 }}>
            Session #{s.id} {stats.student_name ? `· ${stats.student_name}` : ''}
          </div>
          <div className="sub">
            {formatDateTime(s.started_at)} {s.ended_at ? `→ ${formatDateTime(s.ended_at)}` : '(in progress)'}
          </div>
        </div>
        <span className={`badge ${scoreBadge(s.avg_focus_score)}`}>Focus {formatScore(s.avg_focus_score)}</span>
      </div>

      <div className="grid stats">
        <Stat label="Duration" value={formatDuration(s.duration_seconds)} />
        <Stat label="Posture score" value={formatScore(s.posture_score)} hint={s.posture_score == null ? 'Not available' : undefined} />
        <Stat label="Avg distance" value={s.avg_distance_cm == null ? '—' : `${s.avg_distance_cm} cm`} />
        <Stat label="Alerts" value={s.alert_count} />
        <Stat label="Focus min / max" value={f.min == null ? '—' : `${f.min} / ${f.max}`} />
        <Stat label="Behavior logs" value={stats.log_count} />
      </div>

      <div>
        <div className="card-title">Time breakdown</div>
        <BreakdownBar
          formatValue={formatDuration}
          items={[
            { label: 'Focused', value: f.high_seconds, color: 'var(--success)' },
            { label: 'Medium', value: f.medium_seconds, color: 'var(--accent)' },
            { label: 'Distracted', value: f.low_seconds, color: 'var(--warning)' },
            { label: 'Unknown', value: f.unknown_seconds, color: 'var(--unknown)' },
          ]}
        />
        <div className="breakdown-legend">
          <span>Drowsy: <strong>{s.drowsy_seconds == null ? 'Not available' : formatDuration(s.drowsy_seconds)}</strong></span>
          <span>Phone use: <strong>{s.phone_seconds == null ? 'Not available' : formatDuration(s.phone_seconds)}</strong></span>
        </div>
      </div>

      <div className="grid two" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <div>
          <div className="card-title">Gaze direction</div>
          <DistributionRows distribution={stats.gaze_distribution} />
        </div>
        <div>
          <div className="card-title">Presence</div>
          <DistributionRows distribution={stats.presence_distribution} />
        </div>
      </div>

      <div>
        <div className="card-title">Alerts in this session</div>
        {alerts.length === 0 ? (
          <div className="empty"><p>No alerts were raised.</p></div>
        ) : (
          alerts.map((a) => (
            <div className="alert-item" key={a.id}>
              <span className={`badge ${a.severity === 'critical' ? 'badge-crit' : 'badge-warn'}`}>{a.severity}</span>
              <div className="alert-body">
                <div className="alert-title">{a.alert_type}</div>
                <div className="alert-msg">{a.message}</div>
              </div>
              <div className="alert-time">{formatDateTime(a.timestamp)}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
