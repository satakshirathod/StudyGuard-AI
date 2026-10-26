import { useEffect, useState } from 'react';
import { getActiveStudent, getSessions, getWeeklyAnalytics } from '../api/client.js';
import { BarChart, BreakdownBar } from '../components/Charts.jsx';
import SessionDetail from '../components/SessionDetail.jsx';
import { formatDay, formatDateTime, formatDuration } from '../utils/format.js';

export default function Analytics() {
  const [student, setStudent] = useState(null);
  const [days, setDays] = useState(7);
  const [weekly, setWeekly] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const active = await getActiveStudent();
        if (!alive) return;
        setStudent(active);
        if (active) {
          const list = await getSessions(active.id, 50);
          if (!alive) return;
          setSessions(list);
          if (list.length > 0) setSessionId(String(list[0].id));
        }
      } catch (err) {
        if (alive) setError(err.message);
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!student) return undefined;
    let alive = true;
    getWeeklyAnalytics(days)
      .then((data) => alive && setWeekly(data))
      .catch((err) => alive && setError(err.message));
    return () => {
      alive = false;
    };
  }, [student, days]);

  if (loading) return <div className="loading">Loading analytics…</div>;
  if (error) return <div className="card"><div className="form-error">Could not load analytics: {error}</div></div>;

  if (!student) {
    return (
      <div className="card">
        <div className="empty">
          <strong>No active student</strong>
          <p>Create or select a student on the Students page first.</p>
        </div>
      </div>
    );
  }

  const trend = weekly?.days ?? [];
  const chartData = trend.map((d) => ({
    label: formatDay(d.date),
    focus: d.avg_focus_score,
    minutes: Math.round(d.study_seconds / 60),
  }));
  const maxMinutes = Math.max(30, ...chartData.map((d) => d.minutes));
  const totalSeconds = trend.reduce((sum, d) => sum + d.study_seconds, 0);
  const focusedLogs = trend.reduce((sum, d) => sum + d.focused_logs, 0);
  const distractedLogs = trend.reduce((sum, d) => sum + d.distracted_logs, 0);
  const scored = trend.filter((d) => d.avg_focus_score != null);
  const avgFocus = scored.length
    ? Math.round(scored.reduce((sum, d) => sum + d.avg_focus_score, 0) / scored.length)
    : null;

  return (
    <div className="stack">
      <div className="card">
        <div className="row-between" style={{ marginBottom: 16 }}>
          <div className="card-title" style={{ margin: 0 }}>Trends · {student.name}</div>
          <select className="select" value={days} onChange={(e) => setDays(Number(e.target.value))}>
            <option value={7}>Last 7 days</option>
            <option value={14}>Last 14 days</option>
            <option value={30}>Last 30 days</option>
          </select>
        </div>

        <div className="grid stats" style={{ marginBottom: 20 }}>
          <div className="stat"><span className="stat-label">Total study time</span><span className="stat-value">{formatDuration(totalSeconds)}</span></div>
          <div className="stat"><span className="stat-label">Average focus</span><span className="stat-value">{avgFocus ?? '—'}</span></div>
        </div>

        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
          <div>
            <div className="card-title">Focus score per day</div>
            <BarChart data={chartData} valueKey="focus" />
          </div>
          <div>
            <div className="card-title">Study minutes per day</div>
            <BarChart data={chartData} valueKey="minutes" max={maxMinutes} unit="m" />
          </div>
        </div>

        <div style={{ marginTop: 22 }}>
          <div className="card-title">Focused vs distracted (share of recorded behavior)</div>
          <BreakdownBar
            items={[
              { label: 'Focused', value: focusedLogs, color: 'var(--success)' },
              { label: 'Distracted', value: distractedLogs, color: 'var(--warning)' },
            ]}
          />
        </div>
      </div>

      <div className="card">
        <div className="row-between" style={{ marginBottom: 16 }}>
          <div className="card-title" style={{ margin: 0 }}>Individual session analytics</div>
          <select
            className="select"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            disabled={sessions.length === 0}
          >
            {sessions.length === 0 && <option value="">No sessions</option>}
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                #{s.id} · {formatDateTime(s.started_at)}
              </option>
            ))}
          </select>
        </div>
        {sessionId ? (
          <SessionDetail sessionId={Number(sessionId)} />
        ) : (
          <div className="empty">
            <strong>No sessions yet</strong>
            <p>Run a study session from Live Monitoring to see analytics here.</p>
          </div>
        )}
      </div>
    </div>
  );
}
