import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout.jsx';
import Dashboard from './pages/Dashboard.jsx';
import LiveMonitoring from './pages/LiveMonitoring.jsx';
import Analytics from './pages/Analytics.jsx';
import SessionHistory from './pages/SessionHistory.jsx';
import Alerts from './pages/Alerts.jsx';
import Students from './pages/Students.jsx';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/live" element={<LiveMonitoring />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/history" element={<SessionHistory />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/students" element={<Students />} />
      </Routes>
    </Layout>
  );
}