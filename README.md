# 🛡️ StudyGuard AI

> An AI-powered intelligent study monitoring system designed to analyze a student's learning behavior through real-time webcam-based monitoring.

StudyGuard AI is a smart learning-monitoring web application that uses computer vision and AI techniques to monitor a student's study session and analyze different behavioral signals such as face presence, gaze, head pose, attention, drowsiness, posture, phone usage, and screen distance.

The system is designed to generate meaningful study insights, calculate a focus score, provide real-time alerts, and maintain session-based analytics.

---

## 🎯 Project Objective

The main objective of StudyGuard AI is to create an intelligent study environment that can understand a student's learning behavior during a study session.

The system aims to:

- Monitor the student through a webcam.
- Detect whether the student is present.
- Analyze facial landmarks and head movement.
- Estimate gaze direction and attention.
- Detect signs of drowsiness.
- Detect mobile phone usage.
- Analyze sitting posture.
- Estimate screen distance.
- Calculate an overall focus score.
- Generate real-time alerts.
- Store study-session data.
- Provide analytics and session history.

---

# 🚀 Current Project Status

The core live monitoring system is currently functional.

### ✅ Completed

- [x] Web Application Setup
- [x] React + Vite Frontend
- [x] FastAPI Backend
- [x] Frontend–Backend Integration
- [x] Webcam Integration
- [x] Live Camera Feed
- [x] Face Detection
- [x] Face Confidence
- [x] Face Landmarks Detection
- [x] 478-Point Face Landmarks
- [x] Head Pose Detection
- [x] Gaze Detection
- [x] Basic Attention Estimation
- [x] Face Present / Not Detected Status
- [x] Live AI Monitoring Interface
- [x] Backend Health API
- [x] Video/Monitoring Status API

### 🚧 In Development

- [ ] Advanced Gaze Calibration
- [ ] Drowsiness Detection
- [ ] Phone Detection
- [ ] Posture Detection
- [ ] Screen Distance Detection
- [ ] Learning Behavior Analysis
- [ ] Focus Score
- [ ] Real-Time Alerts
- [ ] Study Session Management
- [ ] Database Integration
- [ ] Session Data Storage
- [ ] Session History
- [ ] Student Management
- [ ] Analytics
- [ ] Final Dashboard

---

# 🧠 System Workflow

```text
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
              ├── Face Detection
              │
              ├── Face Landmarks
              │
              ├── Gaze Analysis
              │
              ├── Head Pose
              │
              ├── Attention
              │
              ├── Drowsiness
              │
              ├── Phone Detection
              │
              ├── Posture
              │
              └── Screen Distance
                         │
                         ▼
              Learning Behavior Analysis
                         │
                         ▼
                   Focus Score
                         │
              ┌──────────┴──────────┐
              │                     │
           Alerts               Database
                                    │
                                    ▼
                                Analytics
                                    │
                                    ▼
                             Study Dashboard

                             🏗️ Project Architecture
StudyGuard-AI/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── ...
│   │
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
│   │
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
💻 Technology Stack
Frontend
React
Vite
JavaScript
HTML5
CSS
Web APIs
Webcam / Camera API
Backend
Python
FastAPI
Uvicorn
REST APIs
AI / Computer Vision
MediaPipe
Computer Vision
Face Detection
Face Landmark Detection
Gaze Analysis
Head Pose Estimation
Attention Estimation

Additional AI models for drowsiness, phone detection, posture and other monitoring modules are being integrated.

Database

The project includes a backend database layer for storing study-session and monitoring information.

Database integration and session-level analytics are currently under development.

📸 AI Monitoring Features
1. Webcam Monitoring

The application accesses the user's webcam and provides a real-time camera feed for study-session monitoring.

Current camera testing has been performed using a webcam resolution of approximately:

1280 × 720
15 FPS

The application is designed to work with standard webcams supported by the user's operating system and browser.

2. Face Detection

The system detects whether a student is present in front of the camera.

Example states:

Face Detected
Face Not Detected

Face confidence is also available during detection.

3. Face Landmarks

The current implementation uses facial landmarks for detailed facial analysis.

The system can detect up to:

478 facial landmarks

These landmarks are used as the foundation for gaze, head-pose and other facial-behavior analysis.

4. Head Pose Detection

Head movement/orientation is analyzed to understand the direction in which the student is facing.

The system can use head orientation information for attention analysis.

5. Gaze Detection

The system analyzes eye and facial landmark information to estimate gaze direction.

Example:

LEFT
RIGHT
UP
DOWN
CENTER

Gaze detection is currently implemented and is being refined for more accurate calibration.

6. Attention Estimation

The system provides a basic attention estimation based on available facial and gaze signals.

Example:

Attention: 69%

The attention model will be further refined as additional behavioral signals are integrated.

7. Drowsiness Detection

Drowsiness detection is planned as one of the major AI monitoring modules.

The system will analyze facial/eye-related signals to identify possible signs of:

Sleepiness
Prolonged eye closure
Reduced alertness

Status:

🚧 In Development
8. Phone Detection

Phone detection will identify whether a mobile phone is visible during a study session.

The feature is intended to detect possible distractions caused by mobile-phone usage.

Status:

🚧 In Development
9. Posture Detection

The system will analyze the student's sitting position and posture.

Possible signals include:

Sitting position
Body alignment
Head position
Poor/abnormal posture

Status:

🚧 In Development
10. Screen Distance Detection

The system will estimate whether the student is sitting too close or too far from the screen.

Status:

🚧 In Development
📊 Focus Score

One of the major goals of StudyGuard AI is to generate an overall study focus score.

The final focus score can combine multiple behavioral signals such as:

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
      ↓
   Focus Score

The exact scoring logic will be finalized after all monitoring modules are stable.

Status:

🚧 In Development
🔔 Real-Time Alerts

The system is planned to generate alerts when potentially distracting or abnormal study behavior is detected.

Possible alerts include:

⚠ Student Not Detected

⚠ Looking Away

⚠ Possible Drowsiness

⚠ Phone Detected

⚠ Poor Posture

⚠ Too Close to Screen

⚠ Low Attention

Status:

🚧 In Development
⏱️ Study Session Management

The final system will support complete study sessions.

Expected flow:

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

Each session can eventually contain information such as:

Start time
End time
Duration
Focus score
Attention
Drowsiness events
Phone detection events
Posture events
Away time
Other behavioral events

Status:

🚧 In Development
🗄️ Database

The backend contains a database layer intended to store:

Student information
Study sessions
Behavioral events
Focus metrics
Alerts
Session history

Database integration is currently being developed.

Do not commit personal/student data or secret credentials to the repository.

📈 Analytics Dashboard

The final dashboard will provide a visual representation of study performance.

Planned analytics include:

Study Duration
Focus Score
Attention Percentage
Away Time
Drowsiness Events
Phone Detection Events
Posture Events
Daily/Weekly Trends
Session History

The dashboard will help students understand their study patterns and improve their learning habits.

Status:

🚧 In Development
👨‍🎓 Student Management

The final application is planned to support multiple students/users.

Planned functionality:

Student registration
Student profile
Study sessions
Individual analytics
Session history
Performance tracking

Status:

🚧 In Development
🔌 Backend API

The backend is built using FastAPI.

Current development endpoints include health and video/monitoring status APIs.

Example:

GET /api/health
GET /api/video/status

Development backend:

http://127.0.0.1:8000

FastAPI's interactive API documentation can be accessed during local development at:

http://127.0.0.1:8000/docs
🌐 Frontend

The frontend is developed using React and Vite.

Development frontend:

http://localhost:5173

Live monitoring page:

http://localhost:5173/live
⚙️ Installation & Setup
Prerequisites

Make sure the following are installed:

Python 3.x
Node.js
npm
Git
A modern web browser
Webcam
🐍 Backend Setup

Open the project folder:

cd StudyGuard-AI

Go to backend:

cd backend

Create a virtual environment:

python -m venv .venv
macOS / Linux
source .venv/bin/activate
Windows
.venv\Scripts\activate

Install Python dependencies:

pip install -r requirements.txt

Start the FastAPI server:

uvicorn main:app --reload

Backend should be available at:

http://127.0.0.1:8000
⚛️ Frontend Setup

Open another terminal.

Go to frontend:

cd StudyGuard-AI/frontend

Install dependencies:

npm install

Start the development server:

npm run dev

The frontend should be available at:

http://localhost:5173

Open the live monitoring page:

http://localhost:5173/live
📷 Camera Permission

When opening the live monitoring page, the browser may ask for camera permission.

Select:

Allow Camera Access

Without camera permission, the webcam monitoring features will not work.

🪟 Windows Compatibility

StudyGuard AI is being developed as a cross-platform web application.

The frontend uses browser-based camera access and the backend uses Python/FastAPI.

The application is intended to support:

macOS
Windows
Linux

However, Windows compatibility must be tested with the complete AI model and Python dependency setup before final deployment.

For Windows users:

cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

Then:

cd frontend
npm install
npm run dev
🔒 Security & GitHub Guidelines

Do NOT upload:

.venv/
node_modules/
__pycache__/
*.pyc
.env

Also avoid committing:

API keys
Passwords
Secret tokens
Personal student information
Private database records
Sensitive camera recordings

Example .gitignore:

.venv/
__pycache__/
*.pyc
node_modules/
.env
.DS_Store
dist/
🤝 Team Collaboration

StudyGuard AI is developed as a team project.

To avoid duplicate work:

Inform the team before starting a module.
Work on one assigned module at a time.
Test the implementation properly.
Inform the team when the module is completed.
The next member can then continue from the updated code.
Avoid modifying the same module simultaneously without coordination.
Suggested Work Division
AI Module 1 → Drowsiness Detection

AI Module 2 → Phone Detection

AI Module 3 → Posture Detection

AI Module 4 → Screen Distance

Backend → Session & Database

Frontend → Dashboard & Analytics

Integration → Focus Score & Alerts

The actual division can be updated according to team requirements.

🧪 Testing

The system should be tested for:

Camera
Camera permission
Camera availability
Different resolutions
Different lighting conditions
Face Detection
Face present
Face absent
Multiple faces
Different distances
Gaze
Looking center
Looking left
Looking right
Looking up/down
AI Monitoring
Drowsiness
Phone usage
Posture
Screen distance
Backend
API availability
AI inference
Error handling
Database operations
Frontend
Dashboard
Live monitoring
Alerts
Session history
📌 Development Roadmap
Phase 1 — Project Setup
 Frontend setup
 Backend setup
 API integration
 Project structure
Phase 2 — Webcam & Face Analysis
 Webcam
 Face detection
 Face landmarks
 Head pose
 Gaze
 Basic attention
Phase 3 — Behavioral AI
 Drowsiness
 Phone detection
 Posture
 Screen distance
Phase 4 — Intelligence
 Learning behavior analysis
 Focus score
 Event classification
 Alert generation
Phase 5 — Session Management
 Start study session
 Stop study session
 Session duration
 Session events
 Session storage
Phase 6 — Database
 Database integration
 Student records
 Session records
 Behavior records
 Alert records
Phase 7 — Analytics
 Focus analytics
 Study duration analytics
 Session history
 Student analytics
 Charts and visualizations
Phase 8 — Final Dashboard
 Complete dashboard
 Student management
 Real-time alerts
 Analytics
 Session history
 UI/UX polishing
Phase 9 — Testing & Deployment
 Windows testing
 Cross-browser testing
 AI model testing
 Performance optimization
 Error handling
 Backend deployment
 Frontend deployment
 Final demonstration
📋 Current Development Summary
Working Now
Webcam
   ↓
Face Detection
   ↓
478 Face Landmarks
   ↓
Head Pose
   ↓
Gaze
   ↓
Basic Attention
   ↓
Live Monitoring UI
Next Major Tasks
Drowsiness
     ↓
Phone Detection
     ↓
Posture
     ↓
Screen Distance
     ↓
Focus Score
     ↓
Alerts
     ↓
Study Sessions
     ↓
Database
     ↓
Analytics
     ↓
Final Dashboard
🛠️ Troubleshooting
Camera Not Working

Check:

Browser camera permission.
Another application is not using the camera.
Correct camera is selected.
Backend is running.
Frontend is running.
Backend Not Starting

Check Python environment:

python --version

Activate the virtual environment:

macOS / Linux
source .venv/bin/activate
Windows
.venv\Scripts\activate

Install dependencies again:

pip install -r requirements.txt
Frontend Not Starting

Run:

npm install
npm run dev

Make sure you are inside:

frontend/
AI Model Not Loading

Check:

backend/models/

Make sure the required model files are present and that their paths are correctly configured.

📚 Documentation

Project documentation is available inside:

docs/

Current documentation includes project methodology and related development information.

🎓 Project Type

Major Project

Project: StudyGuard AI

Domain:

Artificial Intelligence
Computer Vision
Machine Learning
Web Development
Learning Analytics
👥 Team
Team Members
Satakshi Rathod — Team Lead
Prakriti Baghel — Team Member
Nidhi Bisen — Team Member
🔮 Future Scope

Future versions of StudyGuard AI can include:

Advanced behavioral analysis
Personalized study recommendations
AI-based productivity reports
Voice-based alerts
Advanced emotion/engagement analysis
Cloud-based analytics
Mobile application
Multi-device support
Teacher/mentor dashboard
Long-term learning behavior analysis
Personalized study plans
⚠️ Disclaimer

StudyGuard AI is an academic/project prototype intended for educational and research purposes.

AI-based behavioral detection can make incorrect predictions depending on lighting, camera quality, user position, occlusion, and other environmental conditions.

The system should therefore be treated as an assistive monitoring and analytics tool rather than a definitive measure of a student's behavior or performance.

⭐ Project Status
🟢 Core Web Application        Working
🟢 Webcam Monitoring           Working
🟢 Face Detection              Working
🟢 Face Landmarks              Working
🟢 Gaze Detection              Working
🟢 Head Pose                   Working
🟢 Basic Attention             Working

🟡 Drowsiness                  In Development
🟡 Phone Detection             In Development
🟡 Posture Detection           In Development
🟡 Screen Distance             In Development
🟡 Focus Score                 In Development
🟡 Alerts                      In Development
🟡 Study Sessions              In Development
🟡 Database                    In Development
🟡 Analytics                   In Development
🟡 Final Dashboard             In Development
🚀 StudyGuard AI

AI-powered study monitoring for smarter and more focused learning.


### Ab ek important correction

Tumhare **current status** ke hisaab se README mein maine ye `[x]` rakha hai:

- Webcam
- Face Detection
- Face Landmarks
- Head Pose
- Gaze
- Basic Attention

Aur jo abhi complete nahi hua hai usko `[ ]`/`In Development` rakha hai. Isse GitHub par **fake completion claim nahi hoga** aur viva/report ke time bhi status clear rahega.

**Ab isko `README.md` mein replace karke commit kar do.**
