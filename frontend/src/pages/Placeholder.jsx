export default function Placeholder({ title, phase, detail }) {
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div className="empty">
        <strong>{phase}</strong>
        <p>{detail}</p>
      </div>
    </div>
  );
}