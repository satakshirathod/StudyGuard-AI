# 🛡️ StudyGuard AI

> **AI-powered intelligent study monitoring system for real-time
> learning behavior analysis.**

StudyGuard AI is a smart learning-monitoring web application that uses
**Computer Vision, Artificial Intelligence, and Learning Analytics** to
analyze a student's study behavior through real-time webcam monitoring.

The system monitors signals such as **face
presence, facial landmarks, head pose, gaze, attention, drowsiness,
phone usage, posture, and screen distance**, and combines
these signals into a **Focus Score**, real-time alerts, session
analytics, and study-history insights.

------------------------------------------------------------------------

## 🎯 Project Objective

The main objective of StudyGuard AI is to create an intelligent study
environment that can understand and analyze a student's learning
behavior during a study session.

### The system aims to:

-   Monitor the student through a webcam
-   Detect whether the student is present
-   Detect facial landmarks
-   Analyze head movement and head pose
-   Estimate gaze direction
-   Estimate basic attention
-   Detect possible drowsiness
-   Detect mobile-phone usage
-   Analyze sitting posture
-   Estimate screen distance
-   Calculate an overall focus score
-   Generate real-time alerts
-   Store study-session data
-   Provide analytics and session history

------------------------------------------------------------------------

# 🚀 Current Project Status

All planned modules are implemented and integrated end-to-end
(webcam → AI analysis → alerts → database → dashboard).

## ✅ Working

  Feature                                        Status
  ---------------------------------------------- ----------
  React + Vite frontend, FastAPI backend         ✅ Working
  Webcam + live camera feed                      ✅ Working
  Face detection + confidence                    ✅ Working
  478-point face landmarks                       ✅ Working
  Head pose (yaw / pitch / roll)                 ✅ Working
  Gaze direction + calibration                   ✅ Working
  Looking outside the screen (left/right/up/down)✅ Working
  Attention estimation + live focus score        ✅ Working
  Drowsiness detection                           ✅ Working
  Phone detection (EfficientDet-Lite2)           ✅ Working
  Posture detection                              ✅ Working
  Screen distance detection                      ✅ Working
  Real-time alerts (saved per session)           ✅ Working
  Study session start / end                      ✅ Working
  SQLite database (students, sessions, logs)     ✅ Working
  Students page (create, list, set active)       ✅ Working
  Session History (list + session details)      ✅ Working
  Analytics (weekly trends + per-session)        ✅ Working
  Alerts page (filter by type / session)         ✅ Working
  Dashboard                                      ✅ Working

## 🔧 Continuous improvement

AI accuracy depends on lighting, camera angle and the person, so the
detection thresholds are meant to be tuned (see
[Tuning & configuration](#-tuning--configuration)).

------------------------------------------------------------------------

# 🧠 System Workflow

``` text
                    START STUDY SESSION
                            │
                            ▼
                       Open Webcam
                            │
                            ▼
                      Detect Student
                            │
                 ┌──────────┴──────────┐
                 │                     │
             Face Found            Face Missing
                 │                     │
                 ▼                     ▼
          Analyze Behavior        Student Away
                 │
     ┌───────────┼─────────────────────────┐
     │           │           │             │
     ▼           ▼           ▼             ▼
 Face        Landmarks      Head          Gaze
Detection                   Pose
     │           │           │             │
     └───────────┴───────────┴─────────────┘
                         │
                         ▼
                 Attention Analysis
                         │
                         ▼
             Behavioral AI Modules
        ┌──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼
   Drowsiness    Phone      Posture   Distance
    Detection   Detection   Detection  Detection
        └──────────┴──────────┴──────────┘
                         │
                         ▼
              Learning Behavior Analysis
                         │
                         ▼
                    Focus Score
                    /        \
                   ▼          ▼
                Alerts     Database
                              │
                              ▼
                          Analytics
                              │
                              ▼
                       Final Dashboard
```

------------------------------------------------------------------------

# 🏗️ Project Architecture

``` text
StudyGuard-AI/
│
├── frontend/
│   ├── src/
│   │   ├── api/            (backend client)
│   │   ├── components/     (charts, session detail)
│   │   ├── pages/          (Dashboard, Live, Analytics, History, Alerts, Students)
│   │   ├── styles/
│   │   └── utils/
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── index.html
│
├── backend/
│   ├── ai/
│   ├── api/
│   ├── database/
│   ├── models/
│   ├── scripts/
│   ├── services/
│   ├── tests/
│   ├── utils/
│   ├── config.py
│   ├── main.py
│   ├── runtime.py
│   ├── requirements.txt
│   └── __init__.py
│
├── docs/
│   └── methodology.md
│
└── README.md
```

------------------------------------------------------------------------

# 💻 Technology Stack

## Frontend

-   React
-   Vite
-   JavaScript
-   HTML5
-   CSS
-   Web APIs
-   Browser Camera / Webcam API

## Backend

-   Python
-   FastAPI
-   Uvicorn
-   REST APIs

## AI / Computer Vision

-   MediaPipe
-   Computer Vision
-   Face Detection
-   Face Landmark Detection
-   Gaze Analysis
-   Head Pose Estimation
-   Attention Estimation

Additional AI models for drowsiness, phone detection, posture, screen
distance, and other monitoring modules are being integrated.

## Database

The backend contains a database layer intended to store:

-   Student information
-   Study sessions
-   Behavioral events
-   Focus metrics
-   Alerts
-   Session history


------------------------------------------------------------------------

# 📸 AI Monitoring Features

## 1. Webcam Monitoring

The application accesses the user's webcam and provides a real-time
camera feed for study-session monitoring.

Current camera testing:

-   Resolution: **1280 × 720**
-   Frame Rate: **15 FPS**

The application is designed to work with standard webcams supported by
the operating system and browser.

------------------------------------------------------------------------

## 2. Face Detection

The system detects whether a student is present in front of the camera.

Example states:

``` text
Face Detected
Face Not Detected
```

Face confidence is also available during detection.

------------------------------------------------------------------------

## 3. Face Landmarks

The current implementation uses facial landmarks for detailed facial
analysis.

The system can detect up to:

**478 facial landmarks**

These landmarks provide the foundation for gaze, head-pose, attention,
and other facial-behavior analysis.

------------------------------------------------------------------------

## 4. Head Pose Detection

Head movement and orientation are analyzed to understand the direction
in which the student is facing.

Head orientation can be used as an input for attention analysis.

------------------------------------------------------------------------

## 5. Gaze Detection

The system analyzes eye and facial landmark information to estimate gaze
direction.

Possible gaze directions include:

-   LEFT
-   RIGHT
-   UP
-   DOWN
-   CENTER

Use the **Calibrate Gaze** button (look straight at the screen) before a
session. Eye movement alone, with the head still facing the screen, is
also detected as looking off-screen.

------------------------------------------------------------------------

## 6. Attention Estimation

The current system provides a basic attention estimation using available
facial, gaze, and head-pose signals.

Example:

``` text
Attention: 69%
```

The attention model will be further refined as additional behavioral
signals are integrated.

------------------------------------------------------------------------

## 7. Drowsiness Detection

Drowsiness detection is one of the major AI monitoring modules under
development.

The system is intended to analyze facial and eye-related signals for
possible signs of:

-   Sleepiness
-   Prolonged eye closure
-   Reduced alertness

**Status:** ✅ Working

------------------------------------------------------------------------

## 8. Phone Detection

Phone detection identifies whether a mobile phone is visible during a
study session, using an on-device object detector
(**EfficientDet-Lite2, int8**, via MediaPipe). A phone that stays in view
for a few seconds raises a **Phone detected** alert.

The model file lives in `backend/models/efficientdet_lite2_int8.tflite`.
A diagnostic tool is available:
`python scripts/phone_debug.py` (stop the backend first, only one
program can use the webcam).

**Status:** ✅ Working

------------------------------------------------------------------------

## 9. Posture Detection

The system will analyze the student's sitting position and posture.

Possible signals include:

-   Sitting position
-   Body alignment
-   Head position
-   Poor or abnormal posture

**Status:** ✅ Working

------------------------------------------------------------------------

## 10. Screen Distance Detection

The system will estimate whether the student is sitting too close to or
too far from the screen.

**Status:** ✅ Working

------------------------------------------------------------------------

# 📊 Focus Score

One of the main goals of StudyGuard AI is to generate an overall **Study
Focus Score**.

The live focus score combines multiple behavioral signals:

``` text
Face Presence
      +
Gaze
      +
Attention
      +
Drowsiness
      +
Posture
      +
Phone Usage
      +
Screen Distance
      │
      ▼
  Focus Score
```

**Status:** ✅ Working

------------------------------------------------------------------------

# 🔔 Real-Time Alerts

The system generates alerts while a study session is active. They are
saved in the database and shown on the **Alerts** page.

Alert types:

-   ⚠️ Student Not Detected
-   ⚠️ Looking Away
-   ⚠️ Possible Drowsiness
-   ⚠️ Phone Detected
-   ⚠️ Poor Posture
-   ⚠️ Too Close to Screen
-   ⚠️ Low Attention

**Status:** ✅ Working

------------------------------------------------------------------------

# ⏱️ Study Session Management

Complete study sessions are supported. Alerts and behavior logs are
recorded only while a session is **active** (press *Start Study Session*).

### Flow

``` text
Start Session
      ↓
Webcam Monitoring
      ↓
AI Behavior Analysis
      ↓
Events & Metrics
      ↓
Focus Score
      ↓
End Session
      ↓
Save Session
      ↓
Generate Analytics
```

A session contains information such as:

-   Start time
-   End time
-   Duration
-   Focus score
-   Attention
-   Drowsiness events
-   Phone detection events
-   Posture events
-   Away time
-   Other behavioral events

**Status:** ✅ Working

------------------------------------------------------------------------

# 🗄️ Database

The database layer stores:

-   Student records
-   Study sessions
-   Behavioral events
-   Focus metrics
-   Alerts
-   Session history

Database: SQLite (file is created automatically and is not committed to Git).

> **Important:** Never commit personal student data, private records,
> passwords, API keys, or other sensitive information to GitHub.

------------------------------------------------------------------------

# 📈 Analytics Dashboard

The dashboard, analytics and session-history pages show study
performance:

-   Study Duration
-   Focus Score
-   Attention Percentage
-   Away Time
-   Drowsiness Events
-   Phone Detection Events
-   Posture Events
-   Daily / Weekly Trends
-   Session History

**Status:** ✅ Working

------------------------------------------------------------------------

# 👨‍🎓 Student Management

The application supports multiple students (one is "active" at a time):

-   Student registration
-   Student profile
-   Study sessions
-   Individual analytics
-   Session history
-   Performance tracking

**Status:** ✅ Working

------------------------------------------------------------------------

# 🔌 Backend API

The backend is built using **FastAPI**. Interactive documentation with
every endpoint is available at `http://127.0.0.1:8000/docs`.

Main endpoints:

``` text
GET  /api/health
GET  /api/video/feed                  live camera stream
GET  /api/video/status                live AI signals
POST /api/ai/gaze/calibrate
POST /api/ai/distance/calibrate

GET  /api/students            POST /api/students
POST /api/students/active     GET  /api/students/active

POST /api/sessions/start      POST /api/sessions/{id}/end
GET  /api/sessions            GET  /api/sessions/active

GET  /api/analytics/sessions/{id}
GET  /api/analytics/weekly?days=7
GET  /api/alerts?session_id=&limit=
GET  /api/dashboard/overview  (+ recent, trend)
```

### Backend

``` text
http://127.0.0.1:8000
```

------------------------------------------------------------------------

# 🌐 Frontend

The frontend is developed using **React + Vite**.

### Development Frontend

``` text
http://localhost:5173
```

### Pages

``` text
/            Dashboard
/live        Live Monitoring
/analytics   Analytics
/history     Session History
/alerts      Alerts
/students    Students
```

------------------------------------------------------------------------

# ⚙️ Installation & Setup

## Prerequisites

Install the following:

-   Python 3.x
-   Node.js
-   npm
-   Git
-   Modern web browser
-   Webcam

------------------------------------------------------------------------

## 🐍 Backend Setup

Open a terminal and go to the project:

``` bash
cd StudyGuard-AI
```

Go to the backend:

``` bash
cd backend
```

Create a virtual environment:

``` bash
python -m venv .venv
```

### macOS / Linux

``` bash
source .venv/bin/activate
```

### Windows

``` bash
.venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Start the FastAPI server:

``` bash
uvicorn main:app --reload
```

Backend:

``` text
http://127.0.0.1:8000
```

------------------------------------------------------------------------

## ⚛️ Frontend Setup

Open another terminal.

From the project directory:

``` bash
cd StudyGuard-AI/frontend
```

Install dependencies:

``` bash
npm install
```

Start the Vite development server:

``` bash
npm run dev
```

Frontend:

``` text
http://localhost:5173
```

Open:

``` text
http://localhost:5173/live
```

------------------------------------------------------------------------

------------------------------------------------------------------------

# 🧪 Run the tests

``` bash
cd backend
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest tests -q
```

------------------------------------------------------------------------

# 🎛️ Tuning & configuration

Settings are read from environment variables (defaults are in
`backend/config.py`). Example for PowerShell, set before `uvicorn`:

``` powershell
$env:SG_PHONE_CONFIDENCE_THRESHOLD="0.30"
$env:SG_GAZE_IRIS_X_THRESHOLD="0.15"
uvicorn main:app --reload
```

  Variable                           Meaning
  ---------------------------------- -------------------------------------------
  `SG_PHONE_DETECTOR_MODEL`          path to the phone detector `.tflite` model
  `SG_PHONE_CONFIDENCE_THRESHOLD`    lower = detects more phones, more false alarms
  `SG_GAZE_IRIS_X_THRESHOLD`         higher = needs a bigger sideways eye movement
  `SG_GAZE_IRIS_Y_THRESHOLD`         higher = needs a bigger up/down eye movement
  `SG_ALERT_PHONE_SECONDS`           seconds a phone must stay visible before an alert
  `SG_ALERT_COOLDOWN_SECONDS`        minimum gap between two alerts of the same type
  `SG_CAMERA_INDEX`                  which webcam to use (default 0)

------------------------------------------------------------------------

# 📷 Camera Permission

When opening the live monitoring page, the browser may request camera
permission.

Select:

``` text
Allow Camera Access
```

Without camera permission, webcam monitoring will not work.

Make sure:

-   The webcam is connected and available
-   No other application is exclusively using the camera
-   The correct camera is selected
-   The frontend is running
-   The backend is running

------------------------------------------------------------------------

# 🪟 Windows Compatibility

StudyGuard AI is being developed as a cross-platform web application.

The architecture is intended to support:

-   macOS
-   Windows
-   Linux

The frontend uses browser-based camera access, while the backend uses
Python/FastAPI.

Windows compatibility should be tested with the complete AI model and
dependency setup before final deployment.

### Windows backend

``` bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Windows frontend

Open another terminal:

``` bash
cd frontend
npm install
npm run dev
```

------------------------------------------------------------------------

# 🔒 Security & GitHub Guidelines

## Do NOT upload

``` text
.venv/
node_modules/
__pycache__/
*.pyc
.env
.DS_Store
dist/
```

Also do not commit:

-   API keys
-   Passwords
-   Secret tokens
-   Personal student information
-   Private database records
-   Sensitive camera recordings

### Recommended `.gitignore`

``` gitignore
.venv/
__pycache__/
*.pyc
node_modules/
.env
.DS_Store
dist/
```

Make sure required AI model files are handled correctly. If large model
files are not committed to the repository, provide clear setup/download
instructions for them.

------------------------------------------------------------------------

# 🤝 Team Collaboration

StudyGuard AI is being developed as a team project.

To avoid duplicate work:

1.  **Inform the team before starting a module.**
2.  **One person should work on one module at a time.**
3.  **Do not modify the same module simultaneously without
    coordination.**
4.  Test your changes properly before handing them over.
5.  Inform the team what was completed.
6.  Mention the files/modules changed.
7.  Mention any remaining issues or dependencies.
8.  Once the work is stable, the next member can continue from the
    updated code.

### Suggested Work Division

  Area          Module
  ------------- --------------------------
  AI Module 1   Drowsiness Detection
  AI Module 2   Phone Detection
  AI Module 3   Posture Detection
  AI Module 4   Screen Distance
  Backend       Study Session & Database
  Frontend      Dashboard & Analytics
  Integration   Focus Score & Alerts

The actual work division can be updated according to team requirements.

> **Team Rule:** If two members work on the same task at the same time
> without informing each other, work may be duplicated or overwritten.
> Always communicate who is continuing the work.

------------------------------------------------------------------------

# 🧪 Testing

The system should be tested across the following areas.

## Camera

-   Camera permission
-   Camera availability
-   Different resolutions
-   Different lighting conditions

## Face Detection

-   Face present
-   Face absent
-   Multiple faces
-   Different distances

## Gaze

-   Looking center
-   Looking left
-   Looking right
-   Looking up
-   Looking down

## AI Monitoring

-   Drowsiness
-   Phone usage
-   Posture
-   Screen distance

## Backend

-   API availability
-   AI inference
-   Error handling
-   Database operations

## Frontend

-   Live monitoring
-   Dashboard
-   Alerts
-   Session history

------------------------------------------------------------------------

# 📌 Development Roadmap

### Phase 1 --- Project Setup

-   Frontend setup
-   Backend setup
-   API integration
-   Project structure

### Phase 2 --- Webcam & Face Analysis

-   Webcam
-   Face detection
-   Face landmarks
-   Head pose
-   Gaze
-   Basic attention

### Phase 3 --- Behavioral AI

-   Drowsiness
-   Phone detection
-   Posture
-   Screen distance

### Phase 4 --- Intelligence

-   Learning behavior analysis
-   Focus score
-   Event classification
-   Alert generation

### Phase 5 --- Session Management

-   Start study session
-   Stop study session
-   Session duration
-   Session events
-   Session storage

### Phase 6 --- Database

-   Database integration
-   Student records
-   Session records
-   Behavior records
-   Alert records

### Phase 7 --- Analytics

-   Focus analytics
-   Study duration analytics
-   Session history
-   Student analytics
-   Charts and visualizations

### Phase 8 --- Final Dashboard

-   Complete dashboard
-   Student management
-   Real-time alerts
-   Analytics
-   Session history
-   UI/UX polishing

### Phase 9 --- Testing & Deployment

-   Windows testing
-   Cross-browser testing
-   AI model testing
-   Performance optimization
-   Error handling
-   Backend deployment
-   Frontend deployment
-   Final demonstration

------------------------------------------------------------------------

# 🛠️ Troubleshooting

## Camera Not Working

Check:

-   Browser camera permission
-   Camera availability
-   Whether another application is using the camera
-   Correct camera selection
-   Backend status
-   Frontend status

## Backend Not Starting

Check Python:

``` bash
python --version
```

Activate the virtual environment.

### macOS / Linux

``` bash
source .venv/bin/activate
```

### Windows

``` bash
.venv\Scripts\activate
```

Then:

``` bash
pip install -r requirements.txt
```

## Frontend Not Starting

Run:

``` bash
npm install
npm run dev
```

Make sure you are inside:

``` text
frontend/
```

## Camera is black / 0 FPS

An old backend process may still be holding the webcam. Stop the backend
(`Ctrl + C`), end leftover `python.exe` / `uvicorn` tasks in Task Manager,
close other apps using the camera, then start the backend again.

## Tests fail with "httpx2 package required"

``` bash
pip install -r requirements.txt
```

## Alerts are not appearing

Alerts are saved only while a study session is active. Press
**Start Study Session**; a phone must stay visible for a few seconds, and
the same alert type has a cooldown.

## AI Model Not Loading

Check:

``` text
backend/models/
```

Make sure:

-   Required model files are present
-   Model paths are correctly configured
-   Required Python dependencies are installed

------------------------------------------------------------------------

# 📚 Documentation

Project documentation is available inside:

``` text
docs/
```

Current documentation includes:

``` text
docs/methodology.md
```

------------------------------------------------------------------------

# 🎓 Project Information

**Project Type:** Major Project

**Project Name:** StudyGuard AI

### Domain

-   Artificial Intelligence
-   Computer Vision
-   Machine Learning
-   Web Development
-   Learning Analytics

------------------------------------------------------------------------

# 👥 Team

  Name                  Role
  --------------------- -------------
  **Satakshi Rathod**   Team Lead
  **Prakriti Baghel**   Team Member
  **Nidhi Bisen**       Team Member

------------------------------------------------------------------------

# 🔮 Future Scope

Future versions of StudyGuard AI can include:

-   Advanced behavioral analysis
-   Personalized study recommendations
-   AI-based productivity reports
-   Voice-based alerts
-   Advanced engagement analysis
-   Cloud-based analytics
-   Mobile application
-   Multi-device support
-   Teacher / mentor dashboard
-   Long-term learning behavior analysis
-   Personalized study plans

------------------------------------------------------------------------

# ⚠️ Disclaimer

StudyGuard AI is an academic/project prototype intended for educational
and research purposes.

AI-based behavioral detection can produce incorrect predictions
depending on factors such as:

-   Lighting conditions
-   Camera quality
-   User position
-   Occlusion
-   Camera angle
-   Environmental conditions

The system should therefore be treated as an **assistive monitoring and
analytics tool**, rather than a definitive measure of a student's
behavior, attention, or performance.

------------------------------------------------------------------------

# ⭐ Project Status

``` text
🟢 Web app (React + FastAPI)    Working
🟢 Webcam, face, landmarks      Working
🟢 Head pose + gaze             Working
🟢 Drowsiness                   Working
🟢 Phone detection              Working
🟢 Posture + screen distance    Working
🟢 Focus score + alerts         Working
🟢 Sessions + database          Working
🟢 Students, history, analytics Working
🟢 Dashboard                    Working
```

------------------------------------------------------------------------

## 🚀 StudyGuard AI

**AI-powered study monitoring for smarter and more focused learning.**
