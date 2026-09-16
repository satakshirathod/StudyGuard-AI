import { useEffect, useState } from 'react';
import { getVideoStatus, gazeCalibrate, distanceCalibrate, VIDEO_FEED_URL } from '../api/client.js';

const GAZE_LABELS = {
  LOOKING_AT_SCREEN: 'AT SCREEN',
  LOOKING_LEFT: 'LEFT',
  LOOKING_RIGHT: 'RIGHT',
  LOOKING_UP: 'UP',
  LOOKING_DOWN: 'DOWN',
  AWAY: 'AWAY',
  UNKNOWN: 'UNKNOWN',
};

function gazeBadge(direction) {
  switch (direction) {
    case 'LOOKING_AT_SCREEN': return 'badge-ok';
    case 'LOOKING_LEFT':
    case 'LOOKING_RIGHT':
    case 'LOOKING_UP':
    case 'LOOKING_DOWN': return 'badge-warn';
    case 'AWAY': return 'badge-crit';
    default: return 'badge-unknown';
  }
}

function drowsinessBadge(state) {
  switch (state) {
    case 'NORMAL': return 'badge-ok';
    case 'DROWSY': return 'badge-crit';
    default: return 'badge-unknown';
  }
}

function distanceBadge(state) {
  switch (state) {
    case 'NORMAL': return 'badge-ok';
    case 'TOO_CLOSE': return 'badge-crit';
    default: return 'badge-unknown';
  }
}

function postureBadge(state) {
  switch (state) {
    case 'GOOD': return 'badge-ok';
    case 'MODERATE': return 'badge-warn';
    case 'POOR': return 'badge-crit';
    default: return 'badge-unknown';
  }
}

function phoneBadge(state) {
  switch (state) {
    case 'PHONE_DETECTED': return 'badge-crit';
    case 'NO_PHONE': return 'badge-ok';
    default: return 'badge-unknown';
  }
}

export default function LiveMonitoring() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [calError, setCalError] = useState(null);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const data = await getVideoStatus();
        if (!alive) return;
        setStatus(data);
        setError(null);
      } catch (err) {
        if (!alive) return;
        setError(err.message || 'Cannot reach the backend');
        setStatus(null);
      }
    };
    poll();
    const id = setInterval(poll, 1500);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const cameraOk = status?.camera_available === true;
  const face = status?.face ?? {};
  const gaze = status?.gaze ?? {};
  const drowsiness = status?.drowsiness ?? {};
  const faceDetected = face.face_detected === true;
  const multiFaces = face.multiple_faces === true;
  const lmOk = face.landmarks_available === true;
  const faceStatus = face.status ?? 'UNKNOWN';
  const dState = drowsiness.state ?? 'UNKNOWN';
  const distance = status?.distance ?? {};
  const distState = distance.state ?? 'UNKNOWN';
  const distCal = distance.calibration_details ?? {};
  const distCalStatus = distCal.status ?? 'REQUIRED';
  const distCalProgress = distCal.required > 0
    ? Math.round((distCal.progress / distCal.required) * 100)
    : 0;
  const distCollecting = distCalStatus === 'COLLECTING';
  const distReady = distCalStatus === 'READY' || distCalStatus === 'CONFIRMED';
  const posture = status?.posture ?? {};
  const pState = posture.state ?? 'UNKNOWN';
  const phone = status?.phone ?? {};
  const phoneState = phone.state ?? 'UNKNOWN';
  const phoneCount = Array.isArray(phone.boxes) ? phone.boxes.length : 0;

  const faceBadge =
    faceStatus === 'DETECTED'
      ? multiFaces ? 'badge-warn' : 'badge-ok'
      : faceStatus === 'NO_FACE' ? 'badge-crit' : 'badge-unknown';
  const faceBadgeText =
    faceStatus === 'DETECTED'
      ? (multiFaces ? `${face.face_count} FACES` : 'DETECTED')
      : faceStatus === 'NO_FACE' ? 'NOT DETECTED' : 'UNKNOWN';

  // Gaze
  const gDirection = gaze.direction ?? 'UNKNOWN';
  const calibration = gaze.calibration ?? {};
  const calStatus = calibration.status ?? 'REQUIRED';
  const calProgress = calibration.required > 0
    ? Math.round((calibration.progress / calibration.required) * 100)
    : 0;
  const collecting = calStatus === 'COLLECTING';
  const calibrated = calStatus === 'READY';

  const handleCalibrate = async () => {
    setCalError(null);
    try {
      await gazeCalibrate(collecting ? 'cancel' : 'start');
    } catch (err) {
      setCalError(err.message || 'Calibration request failed');
    }
  };

  const handleDistCalibrate = async () => {
    setCalError(null);
    try {
      await distanceCalibrate(distCollecting ? 'cancel' : 'start');
    } catch (err) {
      setCalError(err.message || 'Distance calibration request failed');
    }
  };

  const calBadge =
    calStatus === 'READY' ? 'badge-ok'
      : calStatus === 'FAILED' ? 'badge-crit'
        : calStatus === 'COLLECTING' ? 'badge-ok'
          : 'badge-unknown';

  return (
    <div className="live-grid">
      {/* left: video + session controls */}
      <div>
        {cameraOk ? (
          <div className="video-frame">
            <span className="live-tag"><span className="pulse" /> LIVE</span>
            <img src={VIDEO_FEED_URL} alt="StudyGuard live webcam feed" />
            <div className="frame-meta">
              <span>{status.frame_width}×{status.frame_height}</span>
              <span>{status.fps.toFixed(1)} fps</span>
              {faceDetected && (
                <span className="meta-face">
                  {multiFaces ? `${face.face_count} faces` : 'face detected'}
                </span>
              )}
              {gaze.attention_score != null && (
                <span className="meta-face">attention {gaze.attention_score}%</span>
              )}
            </div>
          </div>
        ) : (
          <div className="card error-card">
            <h3>Camera unavailable</h3>
            <div className="sub">{error ? `Backend error: ${error}` : 'The camera could not be opened.'}</div>
            <ol>
              <li>Make sure a webcam is connected and not in use by another app.</li>
              <li>Grant camera permission to the terminal/IDE running the backend
                (macOS: System Settings → Privacy &amp; Security → Camera, or Windows: Settings → Privacy → Camera).</li>
              <li>Restart the backend (<span className="code">uvicorn main:app --reload</span>) after granting permission.</li>
              <li>If you use an external camera, set <span className="code">SG_CAMERA_INDEX=1</span> in <span className="code">.env</span>.</li>
            </ol>
            <div style={{ marginTop: 10 }} className="sub">
              Last error: <span className="code">{status?.last_error || 'not reported'}</span>
            </div>
          </div>
        )}

        <div className="session-actions">
          <button className="btn btn-primary" disabled title="Available from Phase 10 (sessions)">
            Start Study Session
          </button>
          <button className="btn btn-danger" disabled title="Available from Phase 10 (sessions)">
            End Session
          </button>
        </div>
        <div className="footer-note">
          Session controls activate with the database layer (Phase 10). Gaze, drowsiness,
          screen-distance, posture and phone detection are live.
        </div>
      </div>

      {/* right: status panel */}
      <div className="card">
        <h3>Learning Signals</h3>

        {/* Face Presence — real data */}
        <div className="signal-row">
          <div className="signal-name">
            Face
            <small>{faceDetected ? (multiFaces ? `${face.face_count} faces` : '1 face') : 'no face'}</small>
          </div>
          <span className={`badge ${faceBadge}`}>{faceBadgeText}</span>
        </div>

        <div className="signal-row">
          <div className="signal-name">
            Landmarks
            <small>478-point mesh</small>
          </div>
          <span className={`badge ${lmOk ? 'badge-ok' : 'badge-unknown'}`}>
            {faceDetected ? (lmOk ? 'ACTIVE' : 'UNAVAILABLE') : 'WAITING'}
          </span>
        </div>

        {face.confidence_available && faceDetected && (
          <div className="signal-row">
            <div className="signal-name">
              Confidence
              <small>BlazeFace detector</small>
            </div>
            <span className="badge badge-ok">{(face.confidence * 100).toFixed(1)}%</span>
          </div>
        )}

        {/* Gaze — real data (Phase 3) */}
        <h3 style={{ marginTop: 22 }}>Gaze &amp; Attention</h3>

        <div className="signal-row">
          <div className="signal-name">
            Gaze Direction
            <small>iris + head orientation</small>
          </div>
          <span className={`badge ${gazeBadge(gDirection)}`}>{GAZE_LABELS[gDirection] ?? gDirection}</span>
        </div>

        <div className="signal-row">
          <div className="signal-name">
            Attention
            <small>0–100 estimate</small>
          </div>
          <span className={`badge ${gaze.attention_score == null ? 'badge-unknown' : 'badge-ok'}`}>
            {gaze.attention_score == null ? '—' : `${gaze.attention_score}%`}
          </span>
        </div>

        <div className="signal-row">
          <div className="signal-name">
            Confidence
            <small>signal quality</small>
          </div>
          <span className={`badge ${gaze.confidence == null ? 'badge-unknown' : 'badge-ok'}`}>
            {gaze.confidence == null ? '—' : `${(gaze.confidence * 100).toFixed(0)}%`}
          </span>
        </div>

        {gaze.head_yaw_deg != null && (
          <div className="signal-row">
            <div className="signal-name">
              Head Pose
              <small>yaw / pitch / roll</small>
            </div>
            <span className="badge badge-unknown">
              {gaze.head_yaw_deg.toFixed(0)}° / {gaze.head_pitch_deg?.toFixed(0) ?? '—'}° / {gaze.head_roll_deg?.toFixed(0) ?? '—'}°
            </span>
          </div>
        )}

        <div className="signal-row">
          <div className="signal-name">
            Calibration
            <small>resting gaze baseline</small>
          </div>
          <span className={`badge ${calBadge}`}>{calStatus}</span>
        </div>

        <div style={{ marginTop: 6 }}>
          {collecting && (
            <div className="cal-progress">
              <div className="cal-progress-fill" style={{ width: `${calProgress}%` }} />
            </div>
          )}
          <button
            className={`btn ${collecting ? 'btn-danger' : 'btn-ghost'}`}
            style={{ marginTop: 8, width: '100%' }}
            onClick={handleCalibrate}
            disabled={!cameraOk || face.face_detection_enabled === false}
          >
            {collecting
              ? `Cancel (${calibration.progress}/${calibration.required})`
              : calibrated ? 'Recalibrate Gaze' : 'Calibrate Gaze'}
          </button>
          {calStatus === 'FAILED' && calibration.error && (
            <div className="sub" style={{ marginTop: 8 }}>
              <span className="code">{calibration.error}</span>
            </div>
          )}
          {calError && (
            <div className="sub" style={{ marginTop: 8 }}>
              <span className="code">{calError}</span>
            </div>
          )}
          {!cameraOk && (
            <div className="sub" style={{ marginTop: 8 }}>
              Camera must be online before calibrating.
            </div>
          )}
        </div>

        {/* Drowsiness (EAR) — real data (Phase 4) */}
        <h3 style={{ marginTop: 22 }}>Drowsiness</h3>

        <div className="signal-row">
          <div className="signal-name">
            Eye Closure
            <small>eye aspect ratio</small>
          </div>
          <span className={`badge ${drowsinessBadge(dState)}`}>{dState}</span>
        </div>

        {drowsiness.drowsiness_enabled === false && (
          <div className="sub" style={{ marginTop: 6 }}>
            Drowsiness detection is disabled in configuration.
          </div>
        )}

        {drowsiness.ear != null && (
          <div className="signal-row">
            <div className="signal-name">
              EAR
              <small>avg left/right</small>
            </div>
            <span className={`badge ${dState === 'DROWSY' ? 'badge-crit' : 'badge-unknown'}`}>
              {drowsiness.ear.toFixed(3)}
            </span>
          </div>
        )}

        {drowsiness.eyes_closed != null && (
          <div className="signal-row">
            <div className="signal-name">
              Eyes Closed
              <small>this frame</small>
            </div>
            <span className={`badge ${drowsiness.eyes_closed ? 'badge-warn' : 'badge-ok'}`}>
              {drowsiness.eyes_closed ? 'YES' : 'NO'}
            </span>
          </div>
        )}

        {drowsiness.confidence > 0 && (
          <div className="signal-row">
            <div className="signal-name">
              Confidence
              <small>signal quality</small>
            </div>
            <span className="badge badge-unknown">
              {(drowsiness.confidence * 100).toFixed(0)}%
            </span>
          </div>
        )}

        {/* Screen distance — real data (Phase 5) */}
        <h3 style={{ marginTop: 22 }}>Screen Distance</h3>

        <div className="signal-row">
          <div className="signal-name">
            Distance
            <small>relative face width</small>
          </div>
          <span className={`badge ${distanceBadge(distState)}`}>{distState}</span>
        </div>

        {distance.distance_enabled === false && (
          <div className="sub" style={{ marginTop: 6 }}>
            Distance detection is disabled in configuration.
          </div>
        )}

        {distance.face_width != null && (
          <div className="signal-row">
            <div className="signal-name">
              Face Width
              <small>pixels (relative)</small>
            </div>
            <span className="badge badge-unknown">{distance.face_width.toFixed(1)}px</span>
          </div>
        )}

        {distance.distance_estimate != null && (
          <div className="signal-row">
            <div className="signal-name">
              Apparent Size
              <small>{distance.reference_width != null ? `vs ${distance.reference_width.toFixed(0)}px ref` : 'vs reference'}</small>
            </div>
            <span className={`badge ${distState === 'TOO_CLOSE' ? 'badge-crit' : 'badge-unknown'}`}>
              {distance.distance_estimate.toFixed(2)}×
            </span>
          </div>
        )}

        {distance.confidence > 0 && (
          <div className="signal-row">
            <div className="signal-name">
              Confidence
              <small>signal quality</small>
            </div>
            <span className="badge badge-unknown">
              {(distance.confidence * 100).toFixed(0)}%
            </span>
          </div>
        )}

        <div className="signal-row">
          <div className="signal-name">
            Calibration
            <small>reference width</small>
          </div>
          <span className={`badge ${distCalStatus === 'READY' || distCalStatus === 'CONFIRMED' ? 'badge-ok'
            : distCalStatus === 'COLLECTING' || distCalStatus === 'REQUIRED' ? 'badge-warn'
              : distCalStatus === 'DISABLED' ? 'badge-unknown' : 'badge-unknown'}`}>
            {distCalStatus === 'REQUIRED' ? 'REQUIRED' : distCalStatus}
          </span>
        </div>

        <div style={{ marginTop: 6 }}>
          {distCollecting && (
            <div className="cal-progress">
              <div className="cal-progress-fill" style={{ width: `${distCalProgress}%` }} />
            </div>
          )}
          <button
            className={`btn ${distCollecting ? 'btn-danger' : 'btn-ghost'}`}
            style={{ marginTop: 8, width: '100%' }}
            onClick={handleDistCalibrate}
            disabled={!cameraOk || face.face_detection_enabled === false}
          >
            {distCollecting
              ? `Cancel (${distCal.progress}/${distCal.required})`
              : distReady ? 'Recalibrate Distance' : 'Calibrate Distance'}
          </button>
          {distCalStatus === 'REQUIRED' && (
            <div className="sub" style={{ marginTop: 8 }}>
              Sit at your normal working distance, then press Calibrate (or wait — it
              auto-captures from the first stable readings).
            </div>
          )}
        </div>

        {/* Posture (head-pitch) — real data (Phase 6) */}
        <h3 style={{ marginTop: 22 }}>Posture</h3>

        <div className="signal-row">
          <div className="signal-name">
            Sitting Posture
            <small>head-pitch from face landmarks</small>
          </div>
          <span className={`badge ${postureBadge(pState)}`}>{pState}</span>
        </div>

        {posture.posture_enabled === false && (
          <div className="sub" style={{ marginTop: 6 }}>
            Posture detection is disabled in configuration.
          </div>
        )}

        {posture.posture_score != null && (
          <div className="signal-row">
            <div className="signal-name">
              Posture Score
              <small>0–100 estimate</small>
            </div>
            <span className={`badge ${pState === 'POOR' ? 'badge-crit'
              : pState === 'MODERATE' ? 'badge-warn' : 'badge-ok'}`}>
              {posture.posture_score.toFixed(0)}/100
            </span>
          </div>
        )}

        {posture.head_pitch_score != null && (
          <div className="signal-row">
            <div className="signal-name">
              Head Pitch
              <small>nose–eye–chin ratio</small>
            </div>
            <span className="badge badge-unknown">
              {posture.head_pitch_score.toFixed(0)}/100
            </span>
          </div>
        )}

        {posture.face_offset_score != null && (
          <div className="signal-row">
            <div className="signal-name">
              Face Position
              <small>vertical offset</small>
            </div>
            <span className="badge badge-unknown">
              {posture.face_offset_score.toFixed(0)}/100
            </span>
          </div>
        )}

        {posture.confidence > 0 && (
          <div className="signal-row">
            <div className="signal-name">
              Confidence
              <small>signal quality</small>
            </div>
            <span className="badge badge-unknown">
              {(posture.confidence * 100).toFixed(0)}%
            </span>
          </div>
        )}

        {/* Phone detection — real data (Phase 7) */}
        <h3 style={{ marginTop: 22 }}>Phone Detection</h3>

        <div className="signal-row">
          <div className="signal-name">
            Phone in View
            <small>on-device object detector</small>
          </div>
          <span className={`badge ${phoneBadge(phoneState)}`}>
            {phoneState === 'PHONE_DETECTED' ? 'DETECTED'
              : phoneState === 'NO_PHONE' ? 'NOT DETECTED' : 'UNKNOWN'}
          </span>
        </div>

        {phone.phone_enabled === false && (
          <div className="sub" style={{ marginTop: 6 }}>
            Phone detection is disabled in configuration.
          </div>
        )}

        {phone.detector_ready === false && phone.phone_enabled === true && (
          <div className="sub" style={{ marginTop: 6 }}>
            Phone detection model not loaded — run{' '}
            <span className="code">python scripts/download_models.py</span> in the backend.
          </div>
        )}

        {phoneState === 'PHONE_DETECTED' && (
          <>
            <div className="signal-row">
              <div className="signal-name">
                Confidence
                <small>highest phone-box score</small>
              </div>
              <span className="badge badge-crit">
                {(phone.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="signal-row">
              <div className="signal-name">
                Objects
                <small>phone detections this frame</small>
              </div>
              <span className="badge badge-crit">{phoneCount}</span>
            </div>
          </>
        )}

        <h3 style={{ marginTop: 22 }}>Camera</h3>
        <div className="signal-row">
          <div className="signal-name">
            Status
            <small>{cameraOk ? 'streaming' : 'offline'}</small>
          </div>
          <span className={`badge ${cameraOk ? 'badge-ok' : 'badge-crit'}`}>
            {cameraOk ? 'ONLINE' : 'OFFLINE'}
          </span>
        </div>
        {cameraOk && (
          <>
            <div className="signal-row">
              <div className="signal-name">FPS</div>
              <span className="badge badge-ok">{status.fps.toFixed(1)}</span>
            </div>
            <div className="signal-row">
              <div className="signal-name">Resolution</div>
              <span className="badge badge-unknown">
                {status.frame_width}×{status.frame_height}
              </span>
            </div>
          </>
        )}

        {face.status === 'UNKNOWN' && face.error && (
          <div style={{ marginTop: 14 }} className="sub">
            Face module: <span className="code">{face.error}</span>
          </div>
        )}
        {gaze.error && (
          <div style={{ marginTop: 14 }} className="sub">
            Gaze module: <span className="code">{gaze.error}</span>
          </div>
        )}
        {drowsiness.error && (
          <div style={{ marginTop: 14 }} className="sub">
            Drowsiness module: <span className="code">{drowsiness.error}</span>
          </div>
        )}
        {distance.error && (
          <div style={{ marginTop: 14 }} className="sub">
            Distance module: <span className="code">{distance.error}</span>
          </div>
        )}
        {posture.error && (
          <div style={{ marginTop: 14 }} className="sub">
            Posture module: <span className="code">{posture.error}</span>
          </div>
        )}
        {phone.error && (
          <div style={{ marginTop: 14 }} className="sub">
            Phone module: <span className="code">{phone.error}</span>
          </div>
        )}
      </div>
    </div>
  );
}