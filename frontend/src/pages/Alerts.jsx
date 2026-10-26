import { useCallback, useEffect, useState } from 'react';
import { getAlerts, getActiveSession } from '../api/client.js';
import { formatDateTime } from '../utils/format.js';

// alert_type values come from backend/ai/behavior_engine.py
const TYPE_LABELS = {
  STUDENT_AWAY: 'Student away',
  LOOKING_AWAY: 'Looking away',
  DROWSINESS: 'Drowsiness',
  PHONE: 'Phone detected',
  TOO_CLOSE: 'Too close to screen',
  POOR_POSTURE: 'Poor posture',
};

const SEVERITY_BADGE = {
  info: 'badge-unknown',
  warning: 'badge-warn',
  critical: 'badge-crit',
};

export default function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [scope, setScope] = useState('all'); // all | active
  const [type, setType] = useState('ALL');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const session = await getActiveSession();
      setActiveSession(session);
      const sessionId = scope === 'active' && session ? session.id : undefined;
      setAlerts(await getAlerts({ sessionId, limit: 100 }));
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [scope]);

  useEffect(() => {
    load();
    const id = setInterval(load, 5000); // refresh while a session is running
    return () => clearInterval(id);
  }, [load]);

  const visible = type === 'ALL' ? alerts : alerts.filter((a) => a.alert_type === type);

  return (
    <div className="card">
      <div className="row-between" style={{ marginBottom: 14 }}>
        <div className="card-title" style={{ margin: 0 }}>
          Alerts {activeSession && <span className="badge badge-ok" style={{ marginLeft: 8 }}>Session live</span>}
        </div>
        <div className="form-row">
          <select className="select" value={scope} onChange={(e) => setScope(e.target.value)}>
            <option value="all">All sessions</option>
            <option value="active" disabled={!activeSession}>
              Active session only
            </option>
          </select>
          <select className="select" value={type} onChange={(e) => setType(e.target.value)}>
            <option value="ALL">All types</option>
            {Object.entries(TYPE_LABELS).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && <div className="loading">Loading alerts…</div>}
      {!loading && error && <div className="form-error">Could not load alerts: {error}</div>}
      {!loading && !error && visible.length === 0 && (
        <div className="empty">
          <strong>No alerts</strong>
          <p>Alerts appear here when phone use, drowsiness, poor posture or absence is detected during a session.</p>
        </div>
      )}
      {visible.map((a) => (
        <div className="alert-item" key={a.id}>
          <span className={`badge ${SEVERITY_BADGE[a.severity] || 'badge-unknown'}`}>{a.severity}</span>
          <div className="alert-body">
            <div className="alert-title">{TYPE_LABELS[a.alert_type] || a.alert_type}</div>
            <div className="alert-msg">{a.message}</div>
          </div>
          <div className="alert-time">
            {formatDateTime(a.timestamp)}
            <br />
            Session #{a.session_id}
          </div>
        </div>
      ))}
    </div>
  );
}
