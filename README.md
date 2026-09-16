# StudyGuard AI

**Real-Time Student Learning Behavior and Attention Analysis System**

> A real-time AI system that uses a student's webcam to analyze learning behavior
> while studying: face presence, gaze direction, drowsiness, screen distance,
> sitting posture, mobile phone usage, distraction, and an overall AI focus score.

---

## Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Project setup + webcam capture + basic UI | ✅ Done |
| 2 | Face detection + landmarks | ✅ Done |
| 3 | Gaze tracking | ✅ Done |
| 4 | Drowsiness detection | ✅ Done |
| 5 | Distance estimation | ✅ Done |
| 6 | Posture analysis | ✅ Done |
| 7 | Phone detection (YOLO) | ✅ Done |
| 8 | Behavior engine | ✅ Done |
| 9 | Focus score | ✅ Done |
| 10 | SQLite database | ✅ Done |
| 11 | FastAPI endpoints | ✅ Done |
| 12 | React dashboard | ✅ Done |
| 13 | Analytics | ✅ Done |
| 14 | Alerts | ✅ Done |
| 15 | Testing | ⏳ Pending |
| 16 | Documentation | ⏳ Pending |

---

## Phase 1 — Project setup + webcam + basic UI

- **Backend** (FastAPI + OpenCV):
  - Centralised configuration (`backend/config.py`, env-driven via `.env`)
  - Structured logging (`backend/utils/logger.py`)
  - Reusable webcam capture module (`backend/ai/webcam.py`) with graceful
    error handling and rolling FPS tracking
  - MJPEG live video feed at `GET /api/video/feed`
  - Camera status endpoint `GET /api/video/status`
  - Health endpoint `GET /api/health`
- **Frontend** (React + Vite):
  - Professional dark dashboard layout with sidebar + top status bar
  - **Live Monitoring** page showing the real webcam stream, connection status,
    FPS, and the AI-signal panel
  - Placeholder pages for Dashboard, Analytics, Session History, Alerts, Students
  - Empty-state (honest) cards — no fake dashboard values

---

## Phase 2 — Face detection + MediaPipe landmarks

Face detection and 478-point facial landmark extraction are now live in the
video pipeline.

- **`backend/ai/face_detection.py`** — modular `FaceDetectionModule`:
  - **BlazeFace FaceDetector** (`face_detector.tflite`) for real detection
    *confidence* and bounding boxes
  - **MediaPipe FaceLandmarker** (`face_landmarker.task`) for the 478-point
    face mesh
  - Region extraction ready for later phases: left/right eye contours,
    left/right iris, nose, mouth, face outline
  - Primary-face selection (largest bounding box) — designed for **one student**
  - Multiple-face detection is reported explicitly (`multiple_faces=true`);
    downstream use of the primary face only
  - Every failure path returns a structured `FaceResult(status="UNKNOWN")` —
    the server and webcam never crash
- **Backend wiring (`main.py`)**:
  - Models run every `SG_PROCESS_EVERY_N_FRAMES` frames (default 2) to stay real-time
  - Optional dev overlay (`SG_SHOW_FACE_LANDMARKS=true`) draws bbox + landmarks on the feed
  - `GET /api/video/status` now includes a nested `face` block with detection
    state, count, confidence and landmark availability
- **Frontend**: the Live Monitoring page shows real face state —
  `Face: Detected / Not Detected / Multiple Faces`, `Landmarks: Active / Waiting`,
  and detection confidence (when the face detector model is loaded). No fake values.
- **Models**: downloaded by `backend/scripts/download_models.py` into
  `backend/models/` (git-ignored binaries).
- **Tests**: `backend/tests/test_face_detection.py` covers module init, blank
  frames, invalid frames, result structure, overlay drawing and multi-face flags.

> **Honesty note:** this is a computer-vision *estimation*, not a biometric or
> medical assessment. Face-detection confidence comes from the BlazeFace model;
> if that model is unavailable the API reports `confidence_available=false`
> rather than fabricating a value.

### Quick start

```bash
# 1. Download AI models (once)
cd "StudyGuard AI/backend"
python scripts/download_models.py

# 2. Start backend (Python 3.11; using the project venv)
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Start frontend (new terminal)
cd "StudyGuard AI/frontend"
npm install
npm run dev            # → http://localhost:5173 → Live Monitoring
```

Optional: enable the landmark overlay for development.

```bash
echo "SG_SHOW_FACE_LANDMARKS=true" >> "StudyGuard AI/.env"
```

### Run tests

```bash
cd "StudyGuard AI/backend"
.venv/bin/python -m pytest tests/ -v
```

### Manual verification checklist (Phase 2)

1. Open Live Monitoring → webcam streams as before.
2. Sit in front of the camera → `Face: Detected`, `Landmarks: Active`.
3. Turn/look away → `Face: Not Detected`, `Landmarks: Waiting`.
4. Return to camera → detection resumes.
5. (Optional) Test with two people → `N FACES` yellow badge, primary face tracked.

---

## Phase 4 — Real-time drowsiness detection (Eye Aspect Ratio)

Drowsiness is now computed live from the same 478-point MediaPipe face mesh —
no extra camera or model.

- **`backend/ai/drowsiness_detection.py`**:
  - **EAR per eye** (standard metric):
    `EAR = (|p2−p6| + |p3−p5|) / (2 · |p1−p4|)` using the six eye-contour
    points — left eye `[33, 160, 158, 133, 153, 144]`, right eye
    `[362, 385, 387, 263, 373, 380]` — then averaged across usable eyes.
  - **Blink-safe temporal logic**: an eye below `SG_EAR_THRESHOLD` (default
    0.25) starts a consecutive-frame + clock-duration counter; `DROWSY` is only
    reported after the closure persists past `SG_EAR_CONSECUTIVE_FRAMES`
    frames (default 12) *and* `SG_DROWSINESS_DURATION_THRESHOLD` seconds
    (default 0.8s). A short blink resets the counter and never raises DROWSY.
  - Single-eye fallback: if one eye's landmarks are missing/zero-width the
    other eye is used; if neither is usable the reading is honestly `UNKNOWN`
    with numeric measures `None` (nothing is fabricated).
  - States are smoothed with the shared `TemporalStateTracker` (same
    majority-vote utility as gaze); a missing face over time decays to
    `UNKNOWN`.
- **Backend wiring (`main.py`)**:
  - Runs per processed frame after gaze, reusing the existing webcam/face
    pipeline — no second camera, no per-frame DB writes.
  - `GET /api/video/status` now returns a nested `drowsiness` block
    (`state, ear, left_ear, right_ear, eyes_closed, confidence`).
  - The behavior engine consumes drowsiness state in the focus score and the
    alert engine fires a **critical `DROWSINESS` alert** (persisted via the
    alerts table, rate-limited by the shared cooldown) when the stable state is
    `DROWSY`.
- **Frontend**: Live Monitoring shows a **Drowsiness** card (`NORMAL`/`DROWSY`,
  live EAR, per-frame eyes-closed, signal confidence), replacing the old
  Phase-4 placeholder row.
- **Configuration** — see `.env.example`:
  `SG_DROWSINESS_ENABLED`, `SG_SHOW_DROWSINESS_OVERLAY` (dev overlay),
  `SG_EAR_THRESHOLD`, `SG_EAR_CONSECUTIVE_FRAMES`,
  `SG_DROWSINESS_DURATION_THRESHOLD`, `SG_DROWSINESS_SMOOTHING_WINDOW`,
  `SG_DROWSINESS_CHANGE_THRESHOLD`, `SG_ALERT_DROWSINESS_SECONDS`.
- **Tests**: `backend/tests/test_drowsiness_detection.py` covers EAR maths,
  single-eye fallback, missing/zero-width/invalid landmarks, blink vs
  persistent closure, temporal smoothing, honest UNKNOWN, disabled config and
  confidence heuristics + alert integration in `test_behavior_engine.py`.

> **Honesty + disclaimer:** eye-closure estimation depends on camera quality,
> lighting, glasses and occlusion. `SG_EAR_THRESHOLD` is a *tunable estimate* —
> it should be calibrated for the actual student/camera before use. This is an
> approximate computer-vision hint, not a medical or safety diagnostic.

### Manual verification checklist (Phase 4)

1. Start the backend + open Live Monitoring → Drowsiness card shows `NORMAL`.
2. Close your eyes and hold them shut for ~3s → card flips to `DROWSY`.
3. Open your eyes → returns to `NORMAL` after a few frames.
4. Blink normally a few times → state stays `NORMAL` (no false positives).
5. Cover the camera / turn away for several seconds → `UNKNOWN`.
6. (Optional) Set `SG_SHOW_DROWSINESS_OVERLAY=true` → EAR/state chip on feed.

---

## Phase 5 — Real-time screen-distance estimation (relative face width)

Screen distance is estimated live from the same 478-point MediaPipe face mesh —
no extra camera or model — and reported **relatively** (as a width ratio vs a
reference), not as centimetres.

- **`backend/ai/distance_detection.py`**:
  - **Measurement**: pixel distance between the outer eye corners (landmarks
    `33`↔`263`) of the *primary* face — a stable facial-width proxy. Fallback:
    bounding-box width when those landmarks are missing; faces clipped at the
    frame edge, or below `SG_DISTANCE_MIN_FACE_WIDTH` px, are `UNKNOWN`.
  - **Reference/calibration**: `SG_DISTANCE_REFERENCE_WIDTH` (px) when set, or
    **auto-calibration** from the first `SG_DISTANCE_REFERENCE_SAMPLES` (15)
    stable readings — sit at your normal working distance on startup.
    Re-capture anytime via `POST /api/ai/distance/calibrate`. Until a reference
    exists the result is honestly `UNKNOWN`.
  - **Classification**: `ratio = current_width / reference_width`;
    `SG_DISTANCE_TOO_CLOSE_RATIO` (1.45) → `TOO_CLOSE`,
    `SG_DISTANCE_RECOVER_RATIO` (1.25) → `NORMAL`, hysteresis band between them
    holds the current state (no flicker).
  - **Temporal smoothing**: a too-close streak (`SG_DISTANCE_TOO_CLOSE_FRAMES`,
    10) plus the shared majority-vote `TemporalStateTracker`
    (`SG_DISTANCE_SMOOTHING_WINDOW`, 5) prevent noisy-frame flips; a missing
    face over `SG_DISTANCE_AWAY_DURATION` frames decays to `UNKNOWN`.
- **Backend wiring (`main.py`)**:
  - Runs per processed frame after drowsiness in the existing feed — no second
    camera, no second face detector, no per-frame DB writes.
  - `GET /api/video/status` now returns a nested `distance` block
    (`state, distance_estimate, face_width, too_close, confidence,
    reference_width, calibration, error`) plus `distance_enabled` and
    `calibration_details`.
  - New endpoint `POST /api/ai/distance/calibrate` (`start` | `cancel`).
  - The behavior engine consumes distance in the focus score (NORMAL=100,
    TOO_CLOSE=0, UNKNOWN excluded) and the alert engine fires a **warning
    `TOO_CLOSE` alert** ("You are sitting too close to the screen…") only after
    the stable state persists `SG_ALERT_TOO_CLOSE_SECONDS` (5 s), rate-limited
    by the shared cooldown.
- **Frontend**: Live Monitoring shows a **Screen Distance** card
  (`NORMAL`/`TOO_CLOSE`, relative apparent size ×, face width in px, signal
  confidence, calibration status + button), replacing the old Phase-5
  placeholder row. No fake centimetre values are shown.
- **Configuration** — see `.env.example`:
  `SG_DISTANCE_ENABLED`, `SG_SHOW_DISTANCE_OVERLAY`, `SG_DISTANCE_MODE`,
  `SG_DISTANCE_REFERENCE_WIDTH`, `SG_DISTANCE_REFERENCE_SAMPLES`,
  `SG_DISTANCE_TOO_CLOSE_RATIO`, `SG_DISTANCE_RECOVER_RATIO`,
  `SG_DISTANCE_MIN_FACE_WIDTH`, `SG_DISTANCE_TOO_CLOSE_FRAMES`,
  `SG_DISTANCE_SMOOTHING_WINDOW`, `SG_DISTANCE_CHANGE_THRESHOLD`,
  `SG_DISTANCE_AWAY_DURATION`, `SG_ALERT_TOO_CLOSE_SECONDS`.
- **Tests**: `backend/tests/test_distance_detection.py` covers exact-width
  geometry, bbox fallback, clipping, small faces, auto- and manual-calibration,
  calibration cancel/recalibrate, normal/too-close/no-face/small-face states,
  hysteresis, streak debounce, recovery and serialisation; alert + focus-score
  integration lives in `test_behavior_engine.py`.

> **Honesty + disclaimer:** this is an **approximate computer-vision estimate,
> not a calibrated physical measurement**. Without camera calibration the system
> cannot claim exact centimetres — it reports a relative width ratio. Values
> depend on camera position, resolution, lighting and head pose; re-calibrate
> when the camera/seat changes. It is a study-hygiene hint, not a vision-safety
> or ergonomics diagnostic.

### Manual verification checklist (Phase 5)

1. Start the backend + open Live Monitoring while sitting at your normal
   distance → after auto-calibration the Distance card shows `NORMAL`.
2. Move significantly closer to the webcam and hold for a few seconds →
   card flips to `TOO_CLOSE`.
3. Move back to the normal distance → returns to `NORMAL`.
4. Lean forward briefly then back → stays `NORMAL` (no immediate alert/noise).
5. Turn away / cover the camera for several seconds → `UNKNOWN`.
6. (Optional) Set `SG_SHOW_DISTANCE_OVERLAY=true` → distance chip on the feed.

---

## Phase 6 — Real-time sitting-posture analysis (head-pitch signals)

Posture is estimated live from the same 478-point MediaPipe face mesh — no
second camera, no second model — and reported as a 0–100 score classified into
**GOOD** / **MODERATE** / **POOR** / **UNKNOWN**.

- **`backend/ai/posture_detection.py`**:
  - Two complementary **head-pitch signals** computed from the primary face's
    landmarks each processed frame:
    1. **Nose–eye–chin ratio** (landmarks `33`, `263`, `4`, `152`): the nose
       tip's vertical position relative to the eye centre and chin. An upright
       head keeps the nose centred; a slouched head drops the nose lower.
    2. **Face vertical offset**: the face's vertical centre relative to the
       frame — a face positioned unusually low in the frame suggests leaning
       forward.
  - The two signals are blended (65% head-pitch, 35% offset) into a single 0–100
    posture score. Score ≥ `SG_POSTURE_GOOD_THRESHOLD` (70) → **GOOD**,
    ≥ `SG_POSTURE_MODERATE_THRESHOLD` (40) → **MODERATE**, below → **POOR**.
  - **EMA smoothing** (`SG_POSTURE_EMA_ALPHA`, 0.4) smooths the raw score so a
    single jittery frame does not jump the card; the state label passes through
    the shared `TemporalStateTracker` majority vote
    (`SG_POSTURE_SMOOTHING_WINDOW=5`, `SG_POSTURE_CHANGE_THRESHOLD=0.6`).
  - Missing face over `SG_POSTURE_AWAY_DURATION` (10) frames → **UNKNOWN**.
- **Backend wiring (`main.py`)**:
  - Runs per processed frame after distance in the existing feed — no second
    camera, no second model, no per-frame DB writes.
  - `GET /api/video/status` returns a nested `posture` block (`state,
    posture_score, head_pitch_score, face_offset_score, confidence, error,
    posture_enabled`).
  - The behavior engine consumes posture in the focus score: GOOD → 100,
    MODERATE → 50, POOR → 0, UNKNOWN excluded (never fakes a score).
  - The alert engine fires a **warning `POOR_POSTURE` alert** ("Your sitting
    posture looks slouched — try sitting upright for better focus.") when the
    stable state is `POOR` for `SG_ALERT_POOR_POSTURE_SECONDS` (10 s),
    rate-limited by the shared cooldown.
- **Frontend**: Live Monitoring shows a **Posture** card (`GOOD`/`MODERATE`/
  `POOR`, posture score 0–100, head-pitch sub-score, face position sub-score,
  signal confidence) with a config-disabled note when `SG_POSTURE_ENABLED=false`.
- **Configuration** — see `.env.example`:
  `SG_POSTURE_ENABLED`, `SG_SHOW_POSTURE_OVERLAY`, `SG_POSTURE_GOOD_THRESHOLD`,
  `SG_POSTURE_MODERATE_THRESHOLD`, `SG_POSTURE_SMOOTHING_WINDOW`,
  `SG_POSTURE_CHANGE_THRESHOLD`, `SG_POSTURE_EMA_ALPHA`,
  `SG_POSTURE_AWAY_FRAMES`, `SG_ALERT_POOR_POSTURE_SECONDS`.
- **Tests**: `backend/tests/test_posture_detection.py` covers exact head-pitch
  score maths, clamping, degenerate geometry, no/NaN/sparse landmarks, detector
  warmup → GOOD/MODERATE/POOR state transitions, no-face → UNKNOWN, disabled
  config, single-noise robustness, reset, serialisation + alert integration in
  `test_behavior_engine.py`. `backend/tests/test_posture_api.py` covers the
  FastAPI wiring (root phase/modules, status posture block).

> **Honesty + disclaimer:** this is a **head-pitch heuristic** — not a full
> body-posture analysis. A student with good head position but a slouched torso
> may appear `GOOD`, and a forward-leaning student with a tilted camera may appear
> `POOR`. Thresholds depend on camera position and framing; adjust
> `SG_POSTURE_GOOD_THRESHOLD` and `SG_POSTURE_MODERATE_THRESHOLD` for the
> actual setup. It is a study-hygiene hint, not an ergonomic diagnostic.

### Manual verification checklist (Phase 6)

1. Start the backend + open Live Monitoring while sitting upright → Posture
   card shows `GOOD` with score ≈ 70–100.
2. Slouch forward, dropping your head/chin down → card eventually flips to
   `MODERATE` then `POOR`; after `SG_ALERT_POOR_POSTURE_SECONDS` a posture
   alert fires.
3. Sit upright again → card returns to `GOOD`.
4. Cover the camera / turn away for several seconds → `UNKNOWN`.
5. (Optional) Set `SG_SHOW_POSTURE_OVERLAY=true` → posture chip on the feed.

---

## Phase 7 — Real-time phone / mobile-device presence detection

Detects whether a mobile phone is present in the student's visible area using a
YOLO-class on-device object detector and reports it as
**PHONE_DETECTED** / **NO_PHONE** / **UNKNOWN**.

> **Method decision (documented):** the spec labels this phase "Phone detection
> (YOLO)". YOLO via ultralytics would add the heavy torch stack to the project's
> single-dependency (MediaPipe) architecture. We therefore use a **YOLO-class
> object detector from MediaPipe — EfficientDet-Lite0 (COCO)** — through the same
> `mediapipe.tasks` API + `download_models.py` pattern used since Phase 2. It is
> a real frame-level object-detection model (not a heuristic), adds **zero new
> PyPI dependencies**, and COCO's class **67 "cell phone"** directly satisfies
> the spec ("detect a phone in the student's visible area").

- **`backend/ai/phone_detection.py`**:
  - `PhoneObjectDetector` lazily loads `models/efficientdet_lite0.tflite`
    (MediaPipe Tasks, `RunningMode.VIDEO`, per-frame `detect_for_video`).
    Missing model file → honest `UNKNOWN`, never a fabricated `NO_PHONE`.
  - Every detection whose COCO category contains
    `SG_PHONE_LABEL_FILTER` ("phone" → matches "cell phone") and whose score ≥
    `SG_PHONE_CONFIDENCE_THRESHOLD` (0.5) counts as a phone; the highest score
    is reported as `confidence`, the boxes as `boxes`.
  - The raw `PHONE_DETECTED`/`NO_PHONE` state passes through the shared
    `TemporalStateTracker` majority vote (`SG_PHONE_SMOOTHING_WINDOW=5`,
    `SG_PHONE_CHANGE_THRESHOLD=0.6`) so **a single noisy frame never flips the
    card or raises an alert**.
  - Missing/unusable frames over `SG_PHONE_AWAY_FRAMES` (10) → `UNKNOWN`.
- **Backend wiring (`main.py`)**:
  - The detector runs on the whole processed frame (a phone can be visible even
    when `face` tracking is momentarily unavailable) inside the same feed loop.
  - `GET /api/video/status` returns a nested `phone` block (`state, detected,
    confidence, boxes, error, phone_enabled, model_available, detector_ready`).
  - The behavior engine consumes phone presence in the focus score: detected →
    0, not detected → 100, `UNKNOWN` excluded (never fakes a score). Focus
    weights already included `SG_FOCUS_WEIGHT_PHONE=0.05`.
  - The alert engine fires a **warning `PHONE` alert** ("A phone appears to be
    in view — put it down and stay focused.") when the stable state is
    `PHONE_DETECTED` for `SG_ALERT_PHONE_SECONDS` (5 s), rate-limited by the
    shared cooldown. The `PHONE` alert type is no longer pending.
  - `SessionService.record_behavior_log` persists `phone_detected` per row;
    `StudySession.phone_seconds` aggregates positive frames × interval — this
    DB support was already present from the application-foundation phases.
- **Frontend**: Live Monitoring shows a **Phone Detection** card
  (`DETECTED` / `NOT DETECTED` / `UNKNOWN`, confidence + phone-box count when
  detected, model-not-loaded note). "Coming in later phases" is gone — every
  module in the plan is now live.
- **Configuration** — see `.env.example`:
  `SG_PHONE_ENABLED`, `SG_SHOW_PHONE_OVERLAY`, `SG_PHONE_DETECTOR_MODEL`,
  `SG_PHONE_CONFIDENCE_THRESHOLD`, `SG_PHONE_LABEL_FILTER`,
  `SG_PHONE_SMOOTHING_WINDOW`, `SG_PHONE_CHANGE_THRESHOLD`,
  `SG_PHONE_AWAY_FRAMES`, `SG_ALERT_PHONE_SECONDS`.
- **Models**: run `python scripts/download_models.py` — it now also fetches
  EfficientDet-Lite0 (~7 MB) into `backend/models/`.
- **Tests**: `backend/tests/test_phone_detection.py` (uses an injected fake
  object detector — fully deterministic, no real inference needed) covers
  detection, confidence threshold, label filter, `UNKNOWN` degradation, single
  false-positive suppression, persistent detection, recovery, serialisation and
  availability. `test_behavior_engine.py` covers phone focus-score integration +
  `PHONE` alert persistence/cooldown/recovery. `test_phone_api.py` covers the
  FastAPI wiring. Total suite: **193 passed**, frontend build clean.

> **Honesty + disclaimer:** classic object detectors are not perfect — a phone
> that is small, partly occluded or held outside the camera's field of view can
> be missed, and a rare false positive is possible. The confidence threshold +
> temporal smoothing exist to reduce this. If the model file is absent, the
> module reports `UNKNOWN` (it never guesses). No accuracy claim is made without
> a formal evaluation (Phase 15).

### Manual verification checklist (Phase 7)

1. Run `python scripts/download_models.py` in `backend/` to fetch the model.
2. Start the backend + open Live Monitoring → Phone Detection card shows
   `NOT DETECTED` (or `UNKNOWN` if the model did not load — check the note).
3. Hold a phone up in front of the camera → the card flips to `DETECTED` with a
   confidence %; after `SG_ALERT_PHONE_SECONDS` a `PHONE` alert fires.
4. Put the phone away → the card returns to `NOT DETECTED`; no further alerts.
5. (Optional) Set `SG_SHOW_PHONE_OVERLAY=true` → detected-phone boxes + state
   chip drawn on the video feed.
6. Remove/rename `models/efficientdet_lite0.tflite` → card honestly shows
   `UNKNOWN` with a "model not loaded" note instead of a false `NOT DETECTED`.

---

## Project structure (current)

```
StudyGuard AI/
├── .env / .env.example
├── backend/
│   ├── main.py                  # FastAPI app + video feed + AI wiring
│   ├── config.py                # Central configuration
│   ├── requirements.txt
│   ├── ai/
│   │   ├── webcam.py            # Webcam capture (Phase 1)
│   │   ├── face_detection.py    # FaceDetector + FaceLandmarker (Phase 2)
│   │   ├── gaze_tracking.py     # Gaze / attention (Phase 3)
│   │   ├── drowsiness_detection.py  # EAR sleepiness (Phase 4)
│   │   ├── distance_detection.py    # Relative screen distance (Phase 5)
│   │   ├── posture_detection.py     # Sitting-posture head-pitch (Phase 6)
│   │   ├── phone_detection.py       # Phone presence via object detector (Phase 7)
│   │   ├── temporal.py              # Shared state smoothing
│   │   └── behavior_engine.py       # Focus score + alerts (foundation)
│   ├── database/                # SQLAlchemy models + init (foundation)
│   ├── services/                # students / sessions / dashboard / analytics
│   ├── api/                     # FastAPI routers (foundation)
│   ├── runtime.py               # Shared live-state singletons
│   ├── models/                  # Downloaded .task / .tflite models (git-ignored)
│   ├── scripts/
│   │   └── download_models.py
│   ├── tests/
│   │   ├── test_face_detection.py
│   │   ├── test_gaze_tracking.py
│   │   ├── test_drowsiness_detection.py
│   │   ├── test_distance_detection.py
│   │   ├── test_posture_detection.py
│   │   ├── test_posture_api.py
│   │   ├── test_phone_detection.py
│   │   ├── test_phone_api.py
│   │   ├── test_behavior_engine.py
│   │   └── test_session_service.py
│   └── utils/
│       └── logger.py
├── docs/
│   └── methodology.md           # how/why of each phase
└── frontend/
    ├── vite.config.js
    └── src/
        ├── App.jsx / main.jsx / api/client.js
        ├── components/ · styles/ · pages/
        └── pages/LiveMonitoring.jsx   # live face/gaze/drowsiness/posture/phone state
```

Full architecture, methodology, API, and database docs land in Phase 16.