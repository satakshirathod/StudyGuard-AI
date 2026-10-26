import { useEffect, useState } from 'react';
import { getActiveStudent, getSessions } from '../api/client.js';
import SessionDetail from '../components/SessionDetail.jsx';
import { formatDateTime, formatDuration, formatScore, scoreBadge } from '../utils/format.js';

export default function SessionHistory() {
  const [student, setStudent] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const active = await getActiveStudent();
        if (!alive) return;
        setStudent(active);
        if (active) setSessions(await getSessions(active.id, 100));
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

  if (loading) return <div className="loading">Loading sessions…</div>;
  if (error) return <div className="card"><div className="form-error">Could not load sessions: {error}</div></div>;

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

  return (
    <div className="stack">
      <div className="card">
        <div className="card-title">Past sessions · {student.name}</div>
        {sessions.length === 0 ? (
          <div className="empty">
            <strong>No sessions yet</strong>
            <p>Start a session from Live Monitoring. It will show up here after you end it.</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Date</th>
                  <th>Duration</th>
                  <th>Focus score</th>
                  <th>Focused</th>
                  <th>Distracted</th>
                  <th>Alerts</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map((s) => (
                  <tr
                    key={s.id}
                    className={`clickable${s.id === selectedId ? ' selected' : ''}`}
                    onClick={() => setSelectedId(s.id)}
                  >
                    <td>{s.id}</td>
                    <td>{formatDateTime(s.started_at)}</td>
                    <td>{formatDuration(s.duration_seconds)}</td>
                    <td><span className={`badge ${scoreBadge(s.avg_focus_score)}`}>{formatScore(s.avg_focus_score)}</span></td>
                    <td>{formatDuration(s.focused_seconds)}</td>
                    <td>{formatDuration(s.distracted_seconds)}</td>
                    <td>{s.alert_count}</td>
                    <td>
                      <span className={`badge ${s.status === 'active' ? 'badge-ok' : 'badge-unknown'}`}>{s.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {selectedId != null && (
        <div className="card">
          <SessionDetail sessionId={selectedId} />
        </div>
      )}
    </div>
  );
}
