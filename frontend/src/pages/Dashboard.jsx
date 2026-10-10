import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';

import {
  getDashboardOverview,
  getDashboardTrend,
  getDashboardRecent,
  getLiveSessionStatus,
} from '../api/client.js';

function formatMinutes(seconds) {
  if (seconds == null) return '—';

  const minutes = Math.floor(seconds / 60);

  if (minutes < 1) {
    return '< 1';
  }

  return minutes.toString();
}

function formatDistance(value) {
  if (value == null) return '—';

  return Number(value).toFixed(1);
}

function formatDate(value) {
  if (!value) return '—';

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleDateString();
}

function formatDuration(seconds) {
  if (seconds == null) return '—';

  const totalMinutes = Math.floor(seconds / 60);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  return `${minutes}m`;
}

export default function Dashboard() {
  const [overview, setOverview] = useState(null);
  const [live, setLive] = useState(null);
  const [trend, setTrend] = useState([]);
  const [recent, setRecent] = useState([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // -----------------------------------------
  // Load dashboard data
  // -----------------------------------------
  useEffect(() => {
    let alive = true;

    const loadDashboard = async () => {
      try {
        const [
          overviewData,
          trendData,
          recentData,
          liveData,
        ] = await Promise.all([
          getDashboardOverview(),
          getDashboardTrend(7),
          getDashboardRecent(5),
          getLiveSessionStatus(),
        ]);

        if (!alive) return;

        setOverview(overviewData);
        setTrend(Array.isArray(trendData) ? trendData : []);
        setRecent(Array.isArray(recentData) ? recentData : []);
        setLive(liveData);

        setError(null);
      } catch (err) {
        if (!alive) return;

        setError(
          err.message || 'Could not load dashboard data'
        );
      } finally {
        if (alive) {
          setLoading(false);
        }
      }
    };

    loadDashboard();

    // Refresh dashboard data periodically.
    const interval = setInterval(loadDashboard, 3000);

    return () => {
      alive = false;
      clearInterval(interval);
    };
  }, []);

  // -----------------------------------------
  // Derived values
  // -----------------------------------------

  const activeSession =
    live?.session ?? overview?.active_session ?? null;

  const isSessionActive =
    activeSession?.status === 'active';

  const currentFocus =
    live?.focus_score ??
    overview?.avg_focus_score ??
    null;

  const latestCompletedSession =
    recent.find(
      (session) => session.status === 'ended'
    ) ?? null;

  const postureScore =
    latestCompletedSession?.posture_score ?? null;

  const stats = [
    {
      label: 'Current Focus Score',
      value:
        currentFocus != null
          ? Number(currentFocus).toFixed(0)
          : '—',
      unit: '/ 100',
    },
    {
      label: 'Study Duration',
      value: formatMinutes(
        overview?.study_seconds
      ),
      unit: 'min',
    },
    {
      label: 'Focused Time',
      value: formatMinutes(
        overview?.focused_seconds
      ),
      unit: 'min',
    },
    {
      label: 'Distracted Time',
      value: formatMinutes(
        overview?.distracted_seconds
      ),
      unit: 'min',
    },
    {
      label: 'Drowsiness Time',
      value: formatMinutes(
        overview?.drowsy_seconds
      ),
      unit: 'min',
    },
    {
      label: 'Phone Usage',
      value: formatMinutes(
        overview?.phone_seconds
      ),
      unit: 'min',
    },
    {
      label: 'Average Screen Distance',
      value: formatDistance(
        overview?.avg_distance_cm
      ),
      unit: 'cm',
    },
    {
      label: 'Posture Score',
      value:
        postureScore != null
          ? Number(postureScore).toFixed(0)
          : '—',
      unit: '/ 100',
    },
  ];

  const maxStudySeconds = Math.max(
    ...trend.map(
      (item) => item.study_seconds || 0
    ),
    1
  );

  return (
    <div>

      {/* =========================================
          ERROR
          ========================================= */}
      {error && (
        <div
          className="card"
          style={{
            marginBottom: 18,
            border: '1px solid rgba(255, 90, 90, 0.35)',
          }}
        >
          <div className="card-title">
            Dashboard data unavailable
          </div>

          <div className="sub">
            {error}
          </div>
        </div>
      )}

      {/* =========================================
          STAT CARDS
          ========================================= */}
      <div className="grid stats">

        {stats.map((stat) => (
          <div
            className="card stat"
            key={stat.label}
          >
            <div className="stat-label">
              {stat.label}
            </div>

            <div className="stat-value">
              {loading ? '...' : stat.value}
            </div>

            <div className="stat-hint">
              {stat.unit}
              {' · '}

              {stat.label === 'Posture Score' &&
              postureScore == null
                ? 'no completed session data'
                : overview
                  ? 'today'
                  : 'awaiting session data'}
            </div>
          </div>
        ))}

      </div>

      {/* =========================================
          LIVE FOCUS + QUICK ACTIONS
          ========================================= */}
      <div
        className="grid two"
        style={{ marginTop: 18 }}
      >

        {/* Live Focus */}
        <div className="card">

          <div className="card-title">
            Live Focus
          </div>

          {isSessionActive ? (
            <div
              style={{
                padding: '22px 16px',
                textAlign: 'center',
              }}
            >

              <div
                style={{
                  fontSize: 42,
                  fontWeight: 700,
                  marginBottom: 6,
                }}
              >
                {currentFocus != null
                  ? `${Number(currentFocus).toFixed(0)}`
                  : '—'}
              </div>

              <div className="sub">
                Focus Score / 100
              </div>

              <div
                style={{
                  marginTop: 14,
                }}
              >
                <span className="badge badge-ok">
                  SESSION ACTIVE
                </span>
              </div>

              <p
                className="sub"
                style={{ marginTop: 14 }}
              >
                Session #{activeSession.id}
                {' · '}
                live behavior data is being tracked.
              </p>

            </div>
          ) : (
            <div className="empty">

              <strong>
                No active session
              </strong>

              <p>
                Start a study session from{' '}

                <NavLink to="/live">
                  <span
                    style={{
                      color: 'var(--accent)',
                    }}
                  >
                    Live Monitoring
                  </span>
                </NavLink>

                {' '}to see your focus score
                in real time.
              </p>

            </div>
          )}

        </div>

        {/* Quick Actions */}
        <div className="card">

          <div className="card-title">
            Quick Actions
          </div>

          <div
            style={{
              display: 'grid',
              gap: 10,
              marginTop: 14,
            }}
          >

            <NavLink
              to="/live"
              className="btn btn-primary"
              style={{
                textAlign: 'center',
                textDecoration: 'none',
              }}
            >
              {isSessionActive
                ? 'Open Live Monitoring'
                : 'Start Study Session'}
            </NavLink>

            <NavLink
              to="/history"
              className="btn btn-ghost"
              style={{
                textAlign: 'center',
                textDecoration: 'none',
              }}
            >
              View Session History
            </NavLink>

            <NavLink
              to="/analytics"
              className="btn btn-ghost"
              style={{
                textAlign: 'center',
                textDecoration: 'none',
              }}
            >
              View Analytics
            </NavLink>

          </div>

          <div
            className="sub"
            style={{ marginTop: 14 }}
          >
            Today's sessions:{' '}
            {overview?.session_count ?? 0}
            {' · '}
            Alerts:{' '}
            {overview?.alert_count ?? 0}
          </div>

        </div>

      </div>

      {/* =========================================
          TRENDS
          ========================================= */}
      <h3
        style={{
          margin: '26px 0 12px',
        }}
      >
        Trends
      </h3>

      <div
        className="grid two"
        style={{
          gridTemplateColumns: '1fr 1fr',
        }}
      >

        {/* Focus Score Trend */}
        <div className="card">

          <div className="card-title">
            Focus Score Over Time
          </div>

          {trend.length === 0 ? (
            <div
              className="empty"
              style={{
                padding: '22px 16px',
              }}
            >
              <strong>
                No trend data yet
              </strong>

              <p>
                Complete a study session with
                behavior logs to populate this
                chart.
              </p>
            </div>
          ) : (
            <div
              style={{
                padding: '16px 4px',
              }}
            >

              {trend.map((item) => (
                <div
                  key={item.date}
                  style={{
                    marginBottom: 14,
                  }}
                >

                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      marginBottom: 5,
                    }}
                  >
                    <span className="sub">
                      {formatDate(item.date)}
                    </span>

                    <span className="sub">
                      {item.avg_focus_score != null
                        ? `${Number(
                            item.avg_focus_score
                          ).toFixed(0)}/100`
                        : '—'}
                    </span>
                  </div>

                  <div
                    style={{
                      height: 8,
                      borderRadius: 999,
                      background:
                        'rgba(255,255,255,0.08)',
                      overflow: 'hidden',
                    }}
                  >

                    <div
                      style={{
                        height: '100%',
                        width: `${
                          item.avg_focus_score != null
                            ? Math.max(
                                0,
                                Math.min(
                                  100,
                                  item.avg_focus_score
                                )
                              )
                            : 0
                        }%`,
                        background:
                          'var(--accent)',
                        borderRadius: 999,
                      }}
                    />

                  </div>

                </div>
              ))}

            </div>
          )}

        </div>

        {/* Study Duration Trend */}
        <div className="card">

          <div className="card-title">
            Daily Study Duration
          </div>

          {trend.length === 0 ? (
            <div
              className="empty"
              style={{
                padding: '22px 16px',
              }}
            >
              <strong>
                No study data yet
              </strong>

              <p>
                Your daily study duration will
                appear here after sessions are
                recorded.
              </p>
            </div>
          ) : (
            <div
              style={{
                padding: '16px 4px',
              }}
            >

              {trend.map((item) => (
                <div
                  key={item.date}
                  style={{
                    marginBottom: 14,
                  }}
                >

                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      marginBottom: 5,
                    }}
                  >

                    <span className="sub">
                      {formatDate(item.date)}
                    </span>

                    <span className="sub">
                      {formatDuration(
                        item.study_seconds
                      )}
                    </span>

                  </div>

                  <div
                    style={{
                      height: 8,
                      borderRadius: 999,
                      background:
                        'rgba(255,255,255,0.08)',
                      overflow: 'hidden',
                    }}
                  >

                    <div
                      style={{
                        height: '100%',
                        width: `${
                          ((item.study_seconds || 0) /
                            maxStudySeconds) *
                          100
                        }%`,
                        background:
                          'var(--accent)',
                        borderRadius: 999,
                      }}
                    />

                  </div>

                </div>
              ))}

            </div>
          )}

        </div>

        {/* Behavior Distribution */}
        <div className="card">

          <div className="card-title">
            Behavior Distribution
          </div>

          {overview ? (
            <div
              style={{
                padding: '18px 4px',
              }}
            >

              <div
                className="signal-row"
              >
                <div className="signal-name">
                  Focused
                  <small>
                    high-focus logs
                  </small>
                </div>

                <span className="badge badge-ok">
                  {formatMinutes(
                    overview.focused_seconds
                  )}{' '}
                  min
                </span>
              </div>

              <div
                className="signal-row"
              >
                <div className="signal-name">
                  Distracted
                  <small>
                    low-focus logs
                  </small>
                </div>

                <span className="badge badge-warn">
                  {formatMinutes(
                    overview.distracted_seconds
                  )}{' '}
                  min
                </span>
              </div>

              <div
                className="signal-row"
              >
                <div className="signal-name">
                  Drowsy
                  <small>
                    drowsiness duration
                  </small>
                </div>

                <span className="badge badge-crit">
                  {formatMinutes(
                    overview.drowsy_seconds
                  )}{' '}
                  min
                </span>
              </div>

              <div
                className="signal-row"
              >
                <div className="signal-name">
                  Phone
                  <small>
                    phone detected
                  </small>
                </div>

                <span className="badge badge-crit">
                  {formatMinutes(
                    overview.phone_seconds
                  )}{' '}
                  min
                </span>
              </div>

            </div>
          ) : (
            <div className="empty">
              <strong>
                Awaiting session data
              </strong>
            </div>
          )}

        </div>

        {/* Recent Sessions */}
        <div className="card">

          <div className="card-title">
            Recent Sessions
          </div>

          {recent.length === 0 ? (
            <div
              className="empty"
              style={{
                padding: '22px 16px',
              }}
            >
              <strong>
                No sessions yet
              </strong>

              <p>
                Start and complete a study
                session to see it here.
              </p>
            </div>
          ) : (
            <div
              style={{
                padding: '12px 4px',
              }}
            >

              {recent.slice(0, 5).map((session) => (
                <div
                  key={session.id}
                  className="signal-row"
                >

                  <div className="signal-name">

                    Session #{session.id}

                    <small>
                      {formatDate(
                        session.started_at
                      )}
                      {' · '}
                      {formatDuration(
                        session.duration_seconds
                      )}
                    </small>

                  </div>

                  <span
                    className={`badge ${
                      session.status === 'active'
                        ? 'badge-ok'
                        : 'badge-unknown'
                    }`}
                  >
                    {session.status}
                  </span>

                </div>
              ))}

            </div>
          )}

        </div>

      </div>

    </div>
  );
}