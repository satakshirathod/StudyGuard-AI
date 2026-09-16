import { NavLink } from 'react-router-dom';

const STATS = [
  { label: 'Current Focus Score', unit: '/ 100' },
  { label: 'Study Duration', unit: 'min' },
  { label: 'Focused Time', unit: 'min' },
  { label: 'Distracted Time', unit: 'min' },
  { label: 'Drowsiness Time', unit: 'min' },
  { label: 'Phone Usage', unit: 'min' },
  { label: 'Average Screen Distance', unit: 'cm' },
  { label: 'Posture Score', unit: '/ 100' },
];

const CHARTS = [
  { title: 'Focus Score Over Time', phase: 'Phase 13 · Analytics' },
  { title: 'Daily Focus', phase: 'Phase 13 · Analytics' },
  { title: 'Behavior Distribution', phase: 'Phase 13 · Analytics' },
  { title: 'Drowsiness Events', phase: 'Phase 13 · Analytics' },
];

export default function Dashboard() {
  return (
    <div>
      <div className="grid stats">
        {STATS.map((s) => (
          <div className="card stat" key={s.label}>
            <div className="stat-label">{s.label}</div>
            <div className="stat-value">—</div>
            <div className="stat-hint">{s.unit} · awaiting session data</div>
          </div>
        ))}
      </div>

      <div className="grid two" style={{ marginTop: 18 }}>
        <div className="card">
          <div className="card-title">Live Focus</div>
          <div className="empty">
            <strong>No active session</strong>
            <p>
              Start a study session from{' '}
              <NavLink to="/live"><span style={{ color: 'var(--accent)' }}>Live Monitoring</span></NavLink>{' '}
              (sessions arrive in Phase 10) to see your focus score in real time.
            </p>
          </div>
        </div>
        <div className="card">
          <div className="card-title">Quick Actions</div>
          <div className="empty">
            <strong>Nothing yet</strong>
            <p>Session controls, alerts and analytics populate here as later phases land.</p>
          </div>
        </div>
      </div>

      <h3 style={{ margin: '26px 0 12px' }}>Trends</h3>
      <div className="grid two" style={{ gridTemplateColumns: '1fr 1fr' }}>
        {CHARTS.map((c) => (
          <div className="card" key={c.title}>
            <div className="card-title">{c.title}</div>
            <div className="empty" style={{ padding: '22px 16px' }}>
              <strong>Chart placeholder</strong>
              <p>{c.phase} — populated from real session data only.</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}