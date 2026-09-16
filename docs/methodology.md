# Methodology

**StudyGuard AI — real-time student learning-behavior analysis.**
This file documents the *how and why* of each implemented phase. It is a living
document, updated as phases land.

---

## Phase 2 — Face detection and facial landmark extraction

### Why MediaPipe

MediaPipe's face models are small, fast, and run on CPU at interactive rates,
making them a good fit for real-time webcam analysis on a standard laptop:

* **BlazeFace short-range** — a lightweight face detector giving a bounding box
  and a true detection confidence score.
* **FaceLandmarker** — regresses 478 normalised landmarks (face contour, brows,
  eyes, lips) plus iris landmarks, without needing a GPU.
* Both ship as pre-trained models, so no custom training is required for this
  phase — matching the project requirement to keep the model interface modular.

### Version note (MacOS compatibility)

The MediaPipe *Tasks* Python API on macOS had a fatal graph-initialisation
failure (`DrishtiMetalHelper`, `graph_service.h` check failed) in **1.0.1**.
The project therefore pins `mediapipe>=0.10.14,<1.0`, on which both
`FaceDetector` and `FaceLandmarker` (VIDEO running mode) initialise and run
correctly. Verified via a smoke test on 16-Sep-2026.

### Detection pipeline

```
OpenCV BGR frame
      │  cv2.cvtColor → RGB
      ▼
MediaPipe Image (SRGB)
      │
      ├─ FaceDetector.detect_for_video  → bounding boxes + confidence
      └─ FaceLandmarker.detect_for_video → 478 landmarks per face
      │
      ▼
Match faces (nearest bounding-box centre, one-to-one)
      │
      ▼
Build FaceData per face → sort by area → primary = largest
      │
      ▼
FaceResult (status, count, confidence, landmarks, regions)
```

Both models run in **VIDEO mode** to keep a running state while processing frame
history is preserved.

### Why FaceDetector and FaceLandmarker together

| Signal | Source |
|---|---|
| Face presence / count | either model |
| Bounding box | FaceDetector (official), fallback derived from landmarks |
| Detection confidence | FaceDetector only (honest value; 0.0 if model absent) |
| Landmarks (478) | FaceLandmarker |
| Iris, eye, mouth, nose regions | FaceLandmarker landmark indices |

If the FaceDetector fails to load, the module still serves landmarks and reports
`confidence_available=false` — it must never fabricate a confidence value.

### Region landmark indices used

Standard MediaPipe FaceMesh 478-point indices:

| Region | Indices |
|---|---|
| Left eye contour | 33, 160, 158, 133, 153, 144 |
| Right eye contour | 362, 385, 387, 263, 373, 380 |
| Left iris | 468 |
| Right iris | 473 |
| Nose tip | 4 |
| Mouth outer | 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185 |
| Face oval | 36 standard contour indices |

These regions are what later phases (gaze, EAR/drowsiness, distance via face
size) will consume — this phase only prepares the data.

### Single-student assumption

The product targets one student. When multiple faces appear:

1. All faces are detected and counted;
2. `multiple_faces=true` is reported (an explicit warning for users);
3. Only the **primary face (largest bounding-box area)** feeds downstream
   behavioral analysis.

### Input / output contract

**Input** — a BGR `numpy` frame.

**Output** — `FaceResult`:

```python
{
  "face_detected": bool,
  "face_count": int,
  "multiple_faces": bool,
  "primary_face_detected": bool,
  "landmarks_available": bool,
  "confidence": float,            # BlazeFace score; 0.0 when unavailable
  "confidence_available": bool,
  "status": "DETECTED" | "NO_FACE" | "UNKNOWN",
  "faces": [ FaceData ],          # bbox px + normalised, 478 landmarks,
                                  # eyes, iris, nose, mouth, outline
  "error": str | None
}
```

### Real-time considerations

* Detection is run only every `SG_PROCESS_EVERY_N_FRAMES` frames (default 2);
  between runs the last result is reused for UI and overlay.
* Model instances are created once and reused (no per-frame loading).
* The overlay (`SG_SHOW_FACE_LANDMARKS=true`) is for development only and is
  drawn on the live feed, never stored — the project does not save raw video
  by default.

### Limitations

* Face landmark tracking is a **computer-vision estimation**, not biometric or
  medical assessment, and is sensitive to lighting, pose and occlusion.
* No accuracy metric is claimed without evaluation; manual accuracy checks are
  part of the Phase 15 testing checklist.
* With three or more faces, only the configured `SG_MAX_FACES` (default 2) are
  tracked, and analysis stays on the largest.
* Detection confidence is only available when the BlazeFace model is loaded,
  and even then it reflects detector certainty, not student attention.

---

## Phase 4 — Real-time drowsiness detection (Eye Aspect Ratio)

### Why EAR

The Eye Aspect Ratio is a geometry-only eye-closure metric: it needs just six
landmarks per eye and no learned model, so it is fast on CPU and reuses the
existing 478-point face mesh. A normally-open eye sits around EAR ≈
0.30–0.35; a closed eye drops toward ≈ 0.10–0.20, so a single threshold
classifies "open" vs "closed" per frame.

```
EAR = (|p2 − p6| + |p3 − p5|) / (2 · |p1 − p4|)

  p1 ──────── p4  (eye corners, horizontal distance in the denominator)
  p2, p3           upper lid
  p5, p6           lower lid
```

### Landmark indices used

MediaPipe FaceMesh 478-point indices (the same contours already extracted for
gaze, reused here):

| Eye | p1 | p2 | p3 | p4 | p5 | p6 |
|---|---|---|---|---|---|---|
| Left | 33 | 160 | 158 | 133 | 153 | 144 |
| Right | 362 | 385 | 387 | 263 | 373 | 380 |

The detector averages the two usable eyes; when only one eye's contour is
reliable (missing landmarks, zero width, non-finite coords) that eye alone is
used. If neither is usable the frame is `UNKNOWN` with `ear=None` — nothing is
fabricated.

### Blink vs persistent closure

A single frame (or a normal blink) must never be reported as drowsiness. The
module keeps a per-processed-frame closure streak plus a monotonic clock for
the sustained-closure duration:

* `eyes_closed(frame)` ⟺ `avg_ear < SG_EAR_THRESHOLD`.
* Closure is declared **DROWSY** only when the streak reaches
  `SG_EAR_CONSECUTIVE_FRAMES` (default 12) **and** the closure has persisted
  at least `SG_DROWSINESS_DURATION_THRESHOLD` seconds (default 0.8s).
* Reopening the eyes resets the streak; a brief blink therefore never triggers
  DROWSY.

### Temporal smoothing

The final `NORMAL`/`DROWSY`/`UNKNOWN` label passes through the same
`TemporalStateTracker` used by gaze (majority vote over a small window), so a
single stray reading cannot flip the reported state. With no usable face data
for several frames the stable state decays to `UNKNOWN`.

### Focus score and alerts

The behavior engine folds drowsiness into the weighted focus score (weight
gradually via `SG_FOCUS_WEIGHT_DROWSINESS`, default 0.15): `NORMAL` contributes
100, `DROWSY` contributes 0, and `UNKNOWN` contributes nothing. When the
*stable* state is `DROWSY`, the alert engine persists a **critical**
`DROWSINESS` alert (rate-limited by the shared
`SG_ALERT_COOLDOWN_SECONDS`).

### Shared design rules (from the application foundation)

* Modules stay independent — drowsiness never mutates face/gaze state.
* No per-frame database writes: only the periodic behavior snapshot
  (`SG_BEHAVIOR_LOG_INTERVAL_SECONDS`, default 2s) persists `drowsiness_state`
  and derived aggregates (`session.drowsy_seconds`).
* Every failure path returns `UNKNOWN` with `None` numeric values — the video
  loop never crashes on malformed landmarks.

### Limitations (must be documented to users)

* `SG_EAR_THRESHOLD` is a human/camera-dependent *estimate*; it should be
  calibrated per student and per camera before trust. Lighting, glasses,
  occlusion, head pitch and distance all affect the measured EAR.
* This is an approximate computer-vision hint for a study tool — **not** a
  medical or driver-safety diagnostic. No accuracy claim is made without a
  formal evaluation (part of Phase 15's testing checklist).
* Real-camera accuracy testing requires granted camera permission; automated
  tests use geometrically-exact synthetic eyes.
## Phase 5 — Real-time screen-distance estimation (relative face width)

### Why not centimetres?

The camera is **not calibrated** (no known focal length, sensor size, or marker
distance), so converting pixel width → physical distance is impossible without
extra information. Claiming "≈ 30 cm" would be fabricated. Phase 5 therefore
reports a **relative** estimate: *how much closer (or farther) the face appears
than at a reference distance*.

```
ratio = current_facial_width_px / reference_facial_width_px
```

* `ratio ≈ 1.0` → the face occupies ~the same size as at the reference → NORMAL
* `ratio ≥ SG_DISTANCE_TOO_CLOSE_RATIO` (1.45) → much larger → TOO_CLOSE
* `ratio ≤ SG_DISTANCE_RECOVER_RATIO` (1.25) → clearly smaller → back to NORMAL

### Landmark / face measurement used

The **interocular width** — the pixel distance between the outer eye corners
(landmarks `33` and `263`) of the *primary* face, from the existing 478-point
mesh. Eye corners are stable facial landmarks, mirror the classic
interpupillary-distance practice (IPD-based distance estimation), and are only
mildly affected by expressions. Fallback: bounding-box width when those
landmark indices are unavailable. A box clipped at the frame edge (face
partially out of frame) is treated as UNKNOWN.

### Calibration / reference method

Two ways to define the reference width (px):

1. **Auto-calibration (default, `SG_DISTANCE_REFERENCE_WIDTH=0`)** — the
   first `SG_DISTANCE_REFERENCE_SAMPLES` (15) stable readings are averaged at
   startup. **Sit at your normal working distance when the app starts.**
   `POST /api/ai/distance/calibrate` (`{"action":"start"}`) re-captures it.
2. **Manual (`SG_DISTANCE_REFERENCE_WIDTH=<px>`)** — read `face_width` in the
   API at a known-good distance and set it in `.env`. Until a reference exists
   the result is honestly `UNKNOWN` (nothing fabricated).

The value is stored in pixels, so it is tied to a specific camera + resolution.
*Re-calibrate when the camera, resolution, or seat changes.*

### Threshold + temporal smoothing

* `SG_DISTANCE_TOO_CLOSE_RATIO` (1.45) and `SG_DISTANCE_RECOVER_RATIO` (1.25)
  form a **hysteresis band**: ratios between them "hold" the current state, so
  the card does not flicker around the boundary.
* `SG_DISTANCE_TOO_CLOSE_FRAMES` (10) consecutive close frames must occur
  before a raw TOO_CLOSE is emitted (a brief lean-forward stays NORMAL).
* The shared `TemporalStateTracker`
  (`SG_DISTANCE_SMOOTHING_WINDOW=5`, `SG_DISTANCE_CHANGE_THRESHOLD=0.6`)
  majority-votes the final state — a single noisy frame cannot flip it.
* `SG_DISTANCE_AWAY_DURATION` (10) consecutive no-face frames decay to UNKNOWN.
* `SG_DISTANCE_MIN_FACE_WIDTH` (40 px) — below this, landmark quality is
  unreliable so the result is UNKNOWN (also covers "very small / too far").

### Focus score + alerts

* Focus score: NORMAL → 100, TOO_CLOSE → 0, UNKNOWN excluded (never fakes 0).
* Alert `TOO_CLOSE` (warning) fires only after the stable state is TOO_CLOSE
  for `SG_ALERT_TOO_CLOSE_SECONDS` (5 s); the shared persistence + per-type
  cooldown (`SG_ALERT_COOLDOWN_SECONDS`) rate-limits repeats.

### Shared design rules

* Reuses the existing webcam, face pipeline and primary-face selection —
  no second camera, no new MediaPipe model, no per-frame DB writes.
* Module stays independent: distance never mutates face/gaze/drowsiness state.
* Failure/degradation paths return `UNKNOWN` with `None`/`null` numerics.

### Limitations (must be documented to users)

* **Approximate**, relative screen-distance estimation — **not** calibrated
  centimetres. Do not present `distance_estimate` as a physical measurement.
* Sensitive to camera position/resolution: the pixel width changes with
  resolution and framing; a reference captured at 720p is not valid at 1080p.
* Head-pose sensitive: strong yaw shrinks the apparent interocular width
  (can under-estimate closeness); the bbox fallback absorbs some noise but is
  coarser. Extreme pose → UNKNOWN.
* Lighting, occlusion (hair/glasses), and very small faces reduce landmark
  quality → lower confidence or UNKNOWN.
* Intended as a study-hygiene hint (e.g. "move back from the screen"), not a
  medical vision-safety or ergonomics diagnostic. No accuracy claim is made
  without a formal evaluation (Phase 15).

---

## Phase 6 — Real-time sitting-posture analysis (head-pitch signals)

### What "posture" means here (and why head-pitch)

A consumer webcam generally frames the head (and sometimes shoulders) — not the
full torso/back. Accurately classifying *back* posture would require either a
calibrated body-pose model with a full-body framing, or an IMU/sensor worn by
the student, neither of which is available in this phase. Phase 6 therefore
implements an **approximate head-pitch heuristic**: leaning forward (slouching
toward the screen) systematically drops the head and tips it downward relative
to the body, and two geometry-only signals derived from the existing 478-point
face mesh capture that behaviour without any extra model:

1. **Nose–eye–chin ratio** — the nose tip's vertical position between the eye
   centre and the chin. An upright head keeps the nose roughly centred between
   them; a forward-tilted head drives the nose lower toward the chin.

   ```
   ratio = (nose_y − eye_y) / (chin_y − eye_y)
   ```

   Mapped to 0–100: `ratio 0.45 → 100` (upright), `ratio 0.65 → 0` (slouched),
   clamped outside that range.

2. **Face vertical offset** — the face's vertical centre in the frame
   (`bbox_normalized`). A face sitting unusually low in the frame tends to mean
   the student has bent forward toward the camera/screen; mapped 0.35 → 100,
   0.60 → 0.

The two signals are blended (**65% head-pitch, 35% offset**) into one 0–100
posture score, then classified with the configurable thresholds:

```
score ≥ SG_POSTURE_GOOD_THRESHOLD (70)     → GOOD
score ≥ SG_POSTURE_MODERATE_THRESHOLD (40) → MODERATE
score  < SG_POSTURE_MODERATE_THRESHOLD     → POOR
```

### Landmark indices used

| Signal | Landmarks used |
|---|---|
| Head pitch (nose–eye–chin) | eye corners `33`+`263` (averaged), nose tip `4`, chin `152` |
| Face vertical offset | primary face `bbox_normalized` (from `FaceData`) |

### Temporal smoothing

* **EMA of the raw score** (`SG_POSTURE_EMA_ALPHA`, default 0.4) so a single
  jittery frame cannot jump the displayed 0–100 value.
* The state label passes through the same shared `TemporalStateTracker`
  majority-vote used by gaze/drowsiness/distance
  (`SG_POSTURE_SMOOTHING_WINDOW=5`, `SG_POSTURE_CHANGE_THRESHOLD=0.6`) — a brief
  slouch must hold ≈ 3 of the last 5 frames before `POOR` is raised.
* Missing face data for `SG_POSTURE_AWAY_FRAMES` (10) consecutive frames decays
  to `UNKNOWN`; the retained EMA score is still returned (honest, the state is
  what matters to alerts).

### Focus score + alerts

* Focus score: GOOD → 100, MODERATE → 50, POOR → 0, **UNKNOWN excluded** (a
  head-pitch reading we cannot trust must never be scored 0 — the module
  contributes nothing instead).
* Alert `POOR_POSTURE` (warning, "Your sitting posture looks slouched — try
  sitting upright for better focus.") fires only after the *stable* state is
  `POOR` for `SG_ALERT_POOR_POSTURE_SECONDS` (default 10 s), re-using the shared
  persistence + per-type cooldown (`SG_ALERT_COOLDOWN_SECONDS`) machinery.

### Shared design rules

* Reuses the existing webcam, face pipeline and primary-face selection — no
  second camera, no new MediaPipe model, no per-frame DB writes.
* Module stays independent: posture never mutates face/gaze/drowsiness/distance
  state.
* Failure/degradation paths return `UNKNOWN` with `None`/null numerics:
  sparse/NaN landmarks → UNKNOWN, no face → UNKNOWN, `SG_POSTURE_ENABLED=false`
  → UNKNOWN with an explicit "disabled by configuration" error.
* Database: the periodic behavior snapshot stores `posture_state`
  (GOOD/MODERATE/POOR) exactly like `drowsiness_state`; the session aggregate
  (`session.posture_score`) is the mean over the state→score mapping
  (GOOD=100, MODERATE=50, POOR=0). Whole rows/log rows already existed from the
  application foundation, so no schema migration was required.

### Limitations (must be documented to users)

* **Not a full body-posture analysis** — a student with good head position but a
  slouched torso, or habitually hunched shoulders with a level head, may appear
  GOOD. It is an *approximate head-pitch heuristic*.
* Camera position/framing matters: a camera mounted far above or below the head
  biases the face-offset signal; a tilted camera can make an upright student
  look POOR. Re-tune `SG_POSTURE_GOOD_THRESHOLD` /
  `SG_POSTURE_MODERATE_THRESHOLD` for the real setup.
* Head-pitch ratio is compressed when the face is very far (small) or heavily
  occluded (glasses rims, masks, hair) → lower confidence or UNKNOWN.
* It is a study-hygiene hint (e.g. "sit up straighter"), **not** an ergonomic,
  medical, or physiotherapy diagnostic. No accuracy claim is made without a
  formal evaluation (Phase 15).

---

## Phase 7 — Real-time phone / mobile-device presence detection

### Why an on-device object detector (and which one)

The spec's Phase 7 row is labelled "Phone detection (YOLO)". An object detector
is the natural tool here — the task is frame-level detection of a discrete
object ("phone present / not present") from raw pixels, which is exactly what
YOLO-family detectors solve. Three implementation routes were evaluated:

1. **YOLO via ultralytics + torch** — matches the label literally but pulls in
   a ~500 MB torch stack, changing the project's dependency footprint from a
   lightweight MediaPipe-only pipeline to a mixed heavy/light one.
2. **MediaPipe ObjectDetector — EfficientDet-Lite0 (COCO)** — a YOLO-class
   on-device object detector through the *same* `mediapipe.tasks` Tasks API +
   `RunningMode.VIDEO` architecture used since Phase 2. Zero new pip
   dependencies (mediapipe is already pinned `>=0.10.14,<1.0`); COCO class 67
   **"cell phone"** directly matches the requirement.
3. Heuristic (motion/glare/hand-region) — high false-alarm rate; the module
   must never fabricate a positive, so this is rejected outright.

**Choice: option 2** — it is a real object-detection model (not a heuristic),
reuses the project's existing dependency and architecture, and degrades honestly
when the model file is missing (honest `UNKNOWN`, not a fabricated `NO_PHONE`).
The docstring, README, and this methodology document all note the "(YOLO)"
label is satisfied by a YOLO-class detector rather than the specific ultralytics
library.

### Frame-level processing

Every processed frame the `PhoneObjectDetector` converts BGR→RGB, wraps it as a
`mediapipe.Image` (`SRGB`), and passes it to `detect_for_video` (VIDEO mode,
timestamp = `frame_w + frame_h` — same monotonically-unique convention used by
the face detector).

The full frame is passed (not just the face crop) because a phone can be visible
even when no face is currently tracked. The detector returns bounding boxes with
COCO class labels and per-box confidence scores. The phone module filters to
detections whose category name contains
`SG_PHONE_LABEL_FILTER` (default `"phone"` → matches COCO `"cell phone"`), then
keeps only those ≥ `SG_PHONE_CONFIDENCE_THRESHOLD` (default 0.5). The highest
remaining score is reported as `confidence`; the boxes as a serialisable list.

A low-level pre-filter (`score_threshold=0.1` in the MediaPipe options) is set
to drop the detector's near-zero noise detections (hundreds of sub-0.05
detections on garbage frames) early, reducing work. The configurable 0.5
threshold remains the semantic decision boundary.

### Temporal smoothing

Raw `PHONE_DETECTED` / `NO_PHONE` states pass through the same shared
`TemporalStateTracker` majority vote used by every other module
(`SG_PHONE_SMOOTHING_WINDOW=5`, `SG_PHONE_CHANGE_THRESHOLD=0.6`). This means a
single noisy false-positive frame **cannot** flip the card or trigger the alert —
at least 3 of the last 5 frames must agree on `PHONE_DETECTED` before the stable
state changes.

Consecutive unusable frames (None frame, inference error, detector not ready) for
`SG_PHONE_AWAY_FRAMES` (10) produce `UNKNOWN`, matching the other modules' "away"
behaviour.

### Focus score + alerts

* Focus weight: `SG_FOCUS_WEIGHT_PHONE=0.05` (already present in the
  application-foundation config).
* Score mapping: `detected=True` → 0, `detected=False` → 100,
  `detected=None` (UNKNOWN) → **excluded** from the focus-score average entirely.
  An honest "cannot tell" must never lower the score.
* Alert `PHONE` (warning, "A phone appears to be in view — put it down and stay
  focused.") fires only after the *stable* state is `PHONE_DETECTED` for
  `SG_ALERT_PHONE_SECONDS` (5 s), using the shared persistence + per-type
  cooldown (`SG_ALERT_COOLDOWN_SECONDS`). The `PHONE` alert type, which was
  listed as pending since Phase 3, is now fully live.

### Honest degradation

* Model file missing → `PhoneObjectDetector.available = False` → phone card
  shows `UNKNOWN` + a "model not loaded" note. Never `NO_PHONE`.
* Inference exception → caught, logged, returns `UNKNOWN` for that frame.
* `SG_PHONE_ENABLED=false` → returns `UNKNOWN` with an explicit
  "disabled by configuration" error.
* Unknown frame (camera feed returning None) → `UNKNOWN`, streak counts toward
  away-decay.

### Database integration

The DB support for phone was already in place from the application-foundation
phases:

* `BehaviorLog.phone_detected` (nullable boolean) — populated from
  `snapshot.phone.detected` by `SessionService.record_behavior_log`.
* `StudySession.phone_seconds` — aggregated at session end as
  `count(True logs) × behavior_log_interval`.
* Dashboard and analytics services already queried these columns; no schema
  migration was needed.

### Limitations (must be documented to users)

* **Imperfect detection** — classic object detectors can miss a phone that is
  small, partially occluded, or held outside the camera's field of view. A rare
  false positive is also possible. The confidence threshold + temporal smoothing
  exist to reduce both, but neither is eliminated.
* Camera resolution, lighting, and angle affect detection quality. A phone held
  at an unusual angle or covered by a hand may go undetected.
* "No phone visible" does not mean "not using a phone" — the student could be
  using a device below the camera's framing. This module detects *visible*
  phones, not phone usage in general.
* No accuracy claim is made without a formal evaluation (Phase 15).
