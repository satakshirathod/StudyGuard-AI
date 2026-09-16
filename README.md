# 🛡️ StudyGuard AI

> **AI-powered intelligent study monitoring system for real-time
> learning behavior analysis.**

StudyGuard AI is a smart learning-monitoring web application that uses
**Computer Vision, Artificial Intelligence, and Learning Analytics** to
analyze a student's study behavior through real-time webcam monitoring.

The system is being developed to monitor signals such as **face
presence, facial landmarks, head pose, gaze, attention, drowsiness,
phone usage, posture, and screen distance**, and eventually combine
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

The **core live monitoring system is currently functional**.

## ✅ Working

  Feature                              Status
  ------------------------------------ ------------
  Web Application                      ✅ Working
  React + Vite Frontend                ✅ Working
  FastAPI Backend                      ✅ Working
  Frontend--Backend Integration        ✅ Working
  Webcam Integration                   ✅ Working
  Live Camera Feed                     ✅ Working
  Face Detection                       ✅ Working
  Face Confidence                      ✅ Working
  Face Landmarks                       ✅ Working
  478-Point Face Landmarks             ✅ Working
  Head Pose Detection                  ✅ Working
  Gaze Detection                       ✅ Working
  Basic Attention Estimation           ✅ Working
  Face Present / Not Detected Status   ✅ Working
  Live AI Monitoring Interface         ✅ Working
  Backend Health API                   ✅ Working
  Video / Monitoring Status API        ✅ Working

## 🚧 In Development

  Feature                      Status
  ---------------------------- -------------------
  Advanced Gaze Calibration    🚧 In Development
  Drowsiness Detection         🚧 In Development
  Phone Detection              🚧 In Development
  Posture Detection            🚧 In Development
  Screen Distance Detection    🚧 In Development
  Learning Behavior Analysis   🚧 In Development
  Focus Score                  🚧 In Development
  Real-Time Alerts             🚧 In Development
  Study Session Management     🚧 In Development
  Database Integration         🚧 In Development
  Session Data Storage         🚧 In Development
  Session History              🚧 In Development
  Student Management           🚧 In Development
  Analytics                    🚧 In Development
  Final Dashboard              🚧 In Development

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
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── ...
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

Database integration and session-level analytics are currently under
development.

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

Gaze detection is currently implemented and is being refined for better
calibration and accuracy.

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

**Status:** 🚧 In Development

------------------------------------------------------------------------

## 8. Phone Detection

Phone detection is intended to identify whether a mobile phone is
visible during a study session.

The feature will help identify possible distractions caused by
mobile-phone usage.

**Status:** 🚧 In Development

------------------------------------------------------------------------

## 9. Posture Detection

The system will analyze the student's sitting position and posture.

Possible signals include:

-   Sitting position
-   Body alignment
-   Head position
-   Poor or abnormal posture

**Status:** 🚧 In Development

------------------------------------------------------------------------

## 10. Screen Distance Detection

The system will estimate whether the student is sitting too close to or
too far from the screen.

**Status:** 🚧 In Development

------------------------------------------------------------------------

# 📊 Focus Score

One of the main goals of StudyGuard AI is to generate an overall **Study
Focus Score**.

The final score is planned to combine multiple behavioral signals:

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

The exact scoring logic will be finalized after the individual
monitoring modules become stable.

**Status:** 🚧 In Development

------------------------------------------------------------------------

# 🔔 Real-Time Alerts

The system is planned to generate alerts when potentially distracting or
abnormal study behavior is detected.

Possible alerts include:

-   ⚠️ Student Not Detected
-   ⚠️ Looking Away
-   ⚠️ Possible Drowsiness
-   ⚠️ Phone Detected
-   ⚠️ Poor Posture
-   ⚠️ Too Close to Screen
-   ⚠️ Low Attention

**Status:** 🚧 In Development

------------------------------------------------------------------------

# ⏱️ Study Session Management

The final system will support complete study sessions.

### Expected flow

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

A session is expected to contain information such as:

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

**Status:** 🚧 In Development

------------------------------------------------------------------------

# 🗄️ Database

The database layer is intended to store:

-   Student records
-   Study sessions
-   Behavioral events
-   Focus metrics
-   Alerts
-   Session history

Database integration is currently being developed.

> **Important:** Never commit personal student data, private records,
> passwords, API keys, or other sensitive information to GitHub.

------------------------------------------------------------------------

# 📈 Analytics Dashboard

The final dashboard is planned to provide a visual representation of
study performance.

Planned analytics include:

-   Study Duration
-   Focus Score
-   Attention Percentage
-   Away Time
-   Drowsiness Events
-   Phone Detection Events
-   Posture Events
-   Daily / Weekly Trends
-   Session History

**Status:** 🚧 In Development

------------------------------------------------------------------------

# 👨‍🎓 Student Management

The final application is planned to support multiple students / users.

Planned functionality:

-   Student registration
-   Student profile
-   Study sessions
-   Individual analytics
-   Session history
-   Performance tracking

**Status:** 🚧 In Development

------------------------------------------------------------------------

# 🔌 Backend API

The backend is built using **FastAPI**.

Current development endpoints include:

``` text
GET /api/health
GET /api/video/status
```

### Backend

``` text
http://127.0.0.1:8000
```

### FastAPI Documentation

``` text
http://127.0.0.1:8000/docs
```

------------------------------------------------------------------------

# 🌐 Frontend

The frontend is developed using **React + Vite**.

### Development Frontend

``` text
http://localhost:5173
```

### Live Monitoring Page

``` text
http://localhost:5173/live
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
🟢 Core Web Application       Working
🟢 Webcam Monitoring          Working
🟢 Face Detection             Working
🟢 Face Landmarks             Working
🟢 Head Pose                  Working
🟢 Gaze Detection             Working
🟢 Basic Attention            Working

🟡 Drowsiness                 In Development
🟡 Phone Detection            In Development
🟡 Posture Detection          In Development
🟡 Screen Distance            In Development
🟡 Focus Score                In Development
🟡 Alerts                     In Development
🟡 Study Sessions             In Development
🟡 Database                   In Development
🟡 Analytics                 In Development
🟡 Final Dashboard            In Development
```

------------------------------------------------------------------------

## 🚀 StudyGuard AI

**AI-powered study monitoring for smarter and more focused learning.**
