// Small dependency-free charts (plain CSS) so no extra npm package is needed.

export function BarChart({ data, valueKey, labelKey = 'label', max = 100, unit = '', formatValue }) {
  if (!data || data.length === 0) {
    return <div className="empty"><p>No data yet.</p></div>;
  }
  return (
    <div className="bar-chart">
      {data.map((d, i) => {
        const v = d[valueKey];
        const pct = v == null ? 0 : Math.min(100, (v / max) * 100);
        return (
          <div className="bar-col" key={i}>
            <div className="bar-value">{v == null ? '—' : formatValue ? formatValue(v) : `${Math.round(v)}${unit}`}</div>
            <div className="bar-track">
              <div className="bar-fill" style={{ height: `${pct}%` }} />
            </div>
            <div className="bar-label">{d[labelKey]}</div>
          </div>
        );
      })}
    </div>
  );
}

// Horizontal breakdown, e.g. focused / distracted / drowsy.
// items: [{ label, value, color }]  (value = any positive number, shown as share of total)
export function BreakdownBar({ items, formatValue }) {
  const total = items.reduce((sum, it) => sum + (it.value || 0), 0);
  if (total <= 0) {
    return <div className="empty"><p>No behavior data recorded.</p></div>;
  }
  return (
    <div>
      <div className="breakdown">
        {items.map((it) =>
          it.value > 0 ? (
            <div
              key={it.label}
              style={{ width: `${(it.value / total) * 100}%`, background: it.color }}
              title={`${it.label}: ${formatValue ? formatValue(it.value) : it.value}`}
            />
          ) : null
        )}
      </div>
      <div className="breakdown-legend">
        {items.map((it) => (
          <span key={it.label}>
            <i style={{ background: it.color }} />
            {it.label}: <strong>{formatValue ? formatValue(it.value || 0) : it.value || 0}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

// Labelled percentage rows, e.g. gaze distribution { CENTER: 80.5, LEFT: 10 }.
export function DistributionRows({ distribution }) {
  const entries = Object.entries(distribution || {}).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return <div className="empty"><p>No data.</p></div>;
  return (
    <div className="dist-rows">
      {entries.map(([name, pct]) => (
        <div className="dist-row" key={name}>
          <span className="dist-name">{name}</span>
          <div className="dist-track"><div className="dist-fill" style={{ width: `${pct}%` }} /></div>
          <span className="dist-pct">{pct}%</span>
        </div>
      ))}
    </div>
  );
}
