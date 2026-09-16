import { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { getHealth } from '../api/client.js';

const NAV = [
  { to: '/', label: 'Dashboard', icon: '▦' },
  { to: '/live', label: 'Live Monitoring', icon: '◉' },
  { to: '/analytics', label: 'Analytics', icon: '◫' },
  { to: '/history', label: 'Session History', icon: '☰' },
  { to: '/alerts', label: 'Alerts', icon: '⚠' },
  { to: '/students', label: 'Students', icon: '☺' },
];

const TITLES = {
  '/': 'Dashboard',
  '/live': 'Live Monitoring',
  '/analytics': 'Analytics',
  '/history': 'Session History',
  '/alerts': 'Alerts',
  '/students': 'Students',
};

export default function Layout({ children }) {
  const location = useLocation();
  const [backendUp, setBackendUp] = useState(null);

  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const data = await getHealth();
        if (alive) setBackendUp(!!data.status === true);
      } catch {
        if (alive) setBackendUp(false);
      }
    };
    check();
    const id = setInterval(check, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const title = TITLES[location.pathname] ?? 'StudyGuard AI';

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">SG</div>
          <div>
            <div className="brand-name">StudyGuard AI</div>
            <div className="brand-sub">Learning Behavior Analysis</div>
          </div>
        </div>
        <nav className="nav">
          <div className="nav-label">Menu</div>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
            >
              <span aria-hidden>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">Phase 1 · Webcam + UI</div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="page-title">{title}</div>
          <div>
            {backendUp === null && <span className="pill">Connecting…</span>}
            {backendUp === true && (
              <span className="pill">
                <span className="dot ok" /> Backend online
              </span>
            )}
            {backendUp === false && (
              <span className="pill">
                <span className="dot bad" /> Backend offline
              </span>
            )}
          </div>
        </header>
        <main className="content">{children}</main>
      </div>
    </div>
  );
}