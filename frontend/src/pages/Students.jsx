import { useCallback, useEffect, useState } from 'react';
import { getStudents, createStudent, setActiveStudent } from '../api/client.js';
import { formatDateTime } from '../utils/format.js';

export default function Students() {
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState(null);

  const load = useCallback(async () => {
    try {
      setStudents(await getStudents());
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleCreate = async (e) => {
    e.preventDefault();
    setFormError(null);
    if (!name.trim()) {
      setFormError('Name is required');
      return;
    }
    setSaving(true);
    try {
      const created = await createStudent(name.trim(), email.trim());
      // First student ever created becomes the active one automatically.
      if (students.length === 0) await setActiveStudent(created.id);
      setName('');
      setEmail('');
      await load();
    } catch (err) {
      setFormError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleActivate = async (id) => {
    try {
      await setActiveStudent(id);
      await load();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="stack">
      <div className="card">
        <div className="card-title">Add new student</div>
        <form className="form-row" onSubmit={handleCreate}>
          <div className="field">
            <label htmlFor="sname">Name</label>
            <input
              id="sname"
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={120}
              placeholder="e.g. Prakriti"
            />
          </div>
          <div className="field">
            <label htmlFor="semail">Email (optional)</label>
            <input
              id="semail"
              className="input"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="name@example.com"
            />
          </div>
          <button className="btn btn-primary" type="submit" disabled={saving}>
            {saving ? 'Saving…' : 'Create student'}
          </button>
        </form>
        {formError && <div className="form-error">{formError}</div>}
      </div>

      <div className="card">
        <div className="card-title">Students</div>
        {loading && <div className="loading">Loading students…</div>}
        {!loading && error && <div className="form-error">Could not load students: {error}</div>}
        {!loading && !error && students.length === 0 && (
          <div className="empty">
            <strong>No students yet</strong>
            <p>Create a student above. The dashboard and live monitoring need an active student.</p>
          </div>
        )}
        {!loading && students.length > 0 && (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Email</th>
                  <th>Created</th>
                  <th>Status</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {students.map((s) => (
                  <tr key={s.id}>
                    <td>{s.name}</td>
                    <td>{s.email || '—'}</td>
                    <td>{formatDateTime(s.created_at)}</td>
                    <td>
                      {s.is_active ? (
                        <span className="badge badge-ok">Active</span>
                      ) : (
                        <span className="badge badge-unknown">Inactive</span>
                      )}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      {!s.is_active && (
                        <button className="btn btn-ghost" onClick={() => handleActivate(s.id)}>
                          Set active
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
