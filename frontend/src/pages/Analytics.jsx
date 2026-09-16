import Placeholder from './Placeholder.jsx';

export default function Analytics() {
  return (
    <div className="grid two">
      <Placeholder
        title="Daily Analytics"
        phase="Coming in Phase 13"
        detail="Total study duration, focused / distracted time, drowsiness and phone-usage duration, average focus score, average distance and posture statistics."
      />
      <Placeholder
        title="Weekly Analytics"
        phase="Coming in Phase 13"
        detail="Daily focus scores, total study time, distraction / drowsiness / phone-use trends."
      />
    </div>
  );
}