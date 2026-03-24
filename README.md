# 🎭 Moody — Emotion Recognition + Hand Gesture Mouse + Voice Assistant

Moody is a desktop application that uses your webcam to **detect your emotions in real-time**, lets you **control your mouse with hand gestures**, and includes a **voice assistant** you can talk to using the wake word **"Hey Moody"**. Everything runs together from a single app window built with Python and Tkinter.

---

## 📁 Project Structure — Where to Find Everything

| File | What It Does |
|------|-------------|
| `emotion_gesture/fullemotionmodule.py` | **The main application** — combines emotion detection, hand gesture control, voice assistant, user profiles, analytics, and the full UI into one file. This is the file that runs when you launch the app. |
| `emotion_gesture/hand_gesture.py` | **Standalone hand gesture mouse controller** — a separate script you can run on its own to control your mouse using hand gestures (without emotion detection). |
| `emotion_gesture/voice_assistant.py` | **Voice assistant engine** — contains the `MoodyVoiceAssistant` class and the `CommandRegistry` with 100+ voice commands. Imported by the main module. |
| `emotion_gesture/live_emotion_inference.py` | **Feature extraction** — calculates 37 facial features (mouth movement, eyebrow slope, head pose, etc.) from MediaPipe face landmarks. These features are what the ML model uses to predict emotions. |
| `emotion_gesture/advanced_analytics.py` | **Analytics engine** — calculates wellbeing score, productivity score, emotional stability, and generates PDF reports. |
| `emotion_gesture/emotion_recognition_app.py` | **Earlier/simpler version** of the emotion app (without gestures, voice, profiles, or analytics). Kept for reference. |
| `emotion_gesture/model2/` | Contains the trained ML model (`emotion_model.joblib`) and label encoder (`label_encoder.joblib`). |
| `emotion_gesture/user_data/` | Stores per-user emotion logs (`*_emotions.json`) and settings (`*_settings.json`), plus the profiles registry (`profiles.json`). |
| `launcher/common_launcher.py` | **Dashboard launcher** — the login/signup screen with a modern purple gradient UI. Users pick a profile here, then it launches the main emotion app. |
| `launcher/theme_config.py` | Theme configuration — manages color themes across the app. |

---

## 🧠 How the Emotion Model Works (Simple Explanation)

**File:** `emotion_gesture/fullemotionmodule.py` → `predict_emotion_from_frame()` method  
**Feature extraction:** `emotion_gesture/live_emotion_inference.py` → `compute_features()` function

### Step-by-step: What happens every frame

1. **Camera captures a frame** — OpenCV (`cv2.VideoCapture`) grabs a frame from your webcam at roughly 30 FPS.

2. **Face is detected using MediaPipe FaceMesh** — Google's MediaPipe library finds 468 facial landmark points on your face (eyes, nose, mouth, jawline, eyebrows, etc.). Each point has an x, y, z coordinate.

3. **37 facial features are calculated** — The `compute_features()` function in `live_emotion_inference.py` takes those 468 landmarks and calculates meaningful measurements:
   - **Mouth features:** How open is your mouth? Are your lip corners up (smile) or down (frown)? What is the mouth curvature?
   - **Eye features:** How open are your eyes? Is there asymmetry between left and right eyes? (Eye Aspect Ratio)
   - **Eyebrow features:** How high or low are your eyebrows? What angle are they at? Are they furrowed together or raised?
   - **Head pose:** Which direction is your head tilting? (yaw, pitch, roll) — calculated using OpenCV's `solvePnP`.
   - **Other:** Nostril flare, cheek position, jaw width, nose-to-mouth distance, face width-to-height ratio, etc.

4. **The ML model makes a prediction** — These 37 feature values are fed into a pre-trained machine learning model (`emotion_model.joblib`, likely a Random Forest or similar classifier trained with scikit-learn). The model outputs a probability for each of the 7 emotions.

5. **Prediction smoothing** — Instead of showing one noisy frame's result, the app keeps a sliding window of the last 10 predictions and averages them. This makes the displayed emotion stable and not jumpy.

6. **Label mapping** — The model's output label (which might be "anger", "sadness", etc.) is mapped to one of the 7 standard labels: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

7. **UI updates** — The detected emotion, confidence percentage, emoji icon, and suggested actions all update in the UI.

### The 7 Emotions the Model Detects

| Emotion | Emoji | What the Face Looks Like |
|---------|-------|--------------------------|
| Happy | 😊 | Smile, raised cheeks, narrow eyes |
| Sad | 😢 | Drooping mouth corners, lowered eyebrows |
| Angry | 😠 | Furrowed brows, tight lips, tense jaw |
| Fear | 😨 | Wide eyes, raised eyebrows, open mouth |
| Surprise | 😮 | Raised eyebrows, wide open eyes and mouth |
| Disgust | 🤢 | Wrinkled nose, raised upper lip |
| Neutral | 😐 | Relaxed face, no strong expression |

---

## 🖐️ How Hand Gesture Mouse Control Works (Simple Explanation)

**Standalone file:** `emotion_gesture/hand_gesture.py`  
**Integrated class:** `emotion_gesture/fullemotionmodule.py` → `HandGestureController` class

### How it works

The hand gesture system uses **MediaPipe Hands** to detect 21 landmark points on your hand from the webcam. It then checks which fingers are extended or folded to recognize different gestures.

### How finger detection works

Each finger has a **TIP** (fingertip) and a **PIP** (middle joint). The code checks:
- If the tip of a finger is **above** (lower y-value) its PIP joint, the finger is **extended** (pointing up).
- If the tip is **below** its PIP, the finger is **folded** (closed).

For the **thumb**, a special check is done — it needs to be extended and far enough from the index finger.

### Gesture → Action Mapping

| Gesture | What It Does |
|---------|-------------|
| **Show all 5 fingers open** | **Toggle ON/OFF** — Activates or deactivates the virtual mouse. This is like a power switch. There's a 1.5-second cooldown so it doesn't toggle accidentally. |
| **Index finger extended (others closed)** | **Move cursor** — Your index fingertip position maps to the screen position. The code uses **smoothing** (averaging current and previous position) so the cursor doesn't jump around. |
| **Thumb + Index extended (others closed)** | **Left click / Drag** — Quick gesture = click. Hold for more than 0.5 seconds = starts dragging. Release = ends drag or registers click. |
| **Index + Pinky extended (others closed)** | **Right click** (rock sign 🤘) — Hold for 0.3 seconds to trigger. Has a 1-second cooldown to avoid repeated clicks. |
| **Index + Middle + Ring extended (pinky closed)** | **Scroll** — Move your 3 fingers above the screen center to scroll up, below the center to scroll down. Has a deadzone to avoid accidental scrolling. |

### How it connects to the main app

In `fullemotionmodule.py`, the `HandGestureController` class is created inside the main `EmotionRecognitionApp`. When you click **"Enable Gestures"** button (or say "enable mouse" to the voice assistant), it calls `gesture_controller.start(cap)`, which:

1. Starts a **separate background thread** so gesture processing doesn't freeze the UI.
2. Shares the same **camera capture** (`self.cap`) that the emotion detection uses.
3. Runs its own MediaPipe Hands processing loop inside that thread.
4. Uses `pyautogui` to actually move the mouse, click, scroll, etc.

The standalone `hand_gesture.py` does the exact same thing but runs independently with its own camera window — useful for testing gestures without the rest of the app.

---

## 🎤 How the Voice Assistant Works (Simple Explanation)

**File:** `emotion_gesture/voice_assistant.py`

### Architecture

The voice assistant has three main parts:

#### 1. MoodyVoiceAssistant (Engine)
This is the core class that handles:
- **Microphone listening** — Uses the `speech_recognition` library to capture audio from your microphone.
- **Wake word detection** — Listens for "Hey Moody" (or common misheard versions like "Hey Movie", "Hey Modi", "Hey Buddy"). The assistant stays asleep until it hears the wake word.
- **Speech-to-text** — Sends captured audio to Google's Speech Recognition API to convert it to text.
- **Text-to-speech** — Uses `pyttsx3` to speak responses out loud.
- **Auto-sleep** — If you don't say anything for 45 seconds, the assistant goes back to sleep.

#### 2. CommandRegistry (Brain)
This class holds **100+ voice commands** organized into categories:
- **System apps** — Open Notepad, Calculator, File Explorer, Task Manager, Word, Excel, etc.
- **Websites** — Open YouTube, Google, Gmail, GitHub, Netflix, Spotify, Reddit, etc.
- **Search** — "Search for [topic]", "Search YouTube for [topic]"
- **Volume & Media** — Volume up/down/mute, play/pause, next/previous track
- **Window management** — Minimize, maximize, close, snap left/right, alt-tab
- **Keyboard actions** — Copy, paste, undo, redo, type text, save, print
- **Scrolling** — Scroll up/down, page up/down, go to top/bottom
- **Mouse control** — Enable/disable gesture mouse, click, right-click, move cursor
- **System** — Lock screen, brightness, Wi-Fi settings, battery status, time/date
- **Fun** — Tell me a joke, motivate me, set timer

#### 3. Smart Matching System
When you say something, the assistant doesn't need exact words. It uses a multi-step matching process:

1. **Fix misheard words** — Common speech recognition mistakes are auto-corrected (e.g., "minimise" → "minimize", "screen short" → "screenshot", "clothes" → "close").
2. **Strip natural language** — Removes filler words like "can you", "please", "I want to", "could you please" so it finds the core command.
3. **Exact match** — Checks if your cleaned-up words match any registered command.
4. **Alias match** — Each command has multiple aliases (e.g., "open youtube" also matches "go to youtube", "launch youtube", "youtube.com").
5. **Fuzzy match** — If nothing matches exactly, it uses similarity scoring (SequenceMatcher) to find the closest command (threshold: 62% similarity).

### How it connects to the main app

In `fullemotionmodule.py`:
- When you click **"Enable Speech"** button, it opens a **Voice Assistant tab** inside the main window.
- Clicking **"Start Listening"** creates a `MoodyVoiceAssistant` instance with callbacks that update the UI log and status labels.
- The assistant runs in a **background thread** so the UI stays responsive.
- The voice assistant can **toggle gesture control** via a callback — saying "enable mouse" calls the same `toggle_gesture_control()` method as clicking the button.
- There is a **command reference panel** on the right side of the voice tab showing all available commands.

---

## 🔗 How Everything Connects Together

```
┌─────────────────────────────────────────────────────────┐
│                  common_launcher.py                      │
│           (Login / Signup Dashboard)                     │
│    Sets MOODY_USER env variable → launches main app      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               fullemotionmodule.py                       │
│           (Main Application Window)                      │
│                                                         │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Camera Feed   │  │  Emotion     │  │  Suggested   │ │
│  │  (OpenCV +     │  │  Display     │  │  Actions     │ │
│  │   MediaPipe    │  │  (Emoji +    │  │  (Buttons    │ │
│  │   FaceMesh)    │  │   Label +    │  │   per mood)  │ │
│  │               │  │   Confidence)│  │              │ │
│  └───────┬───────┘  └──────────────┘  └──────────────┘ │
│          │                                              │
│          │  Shares camera (self.cap)                    │
│          ▼                                              │
│  ┌───────────────────────┐  ┌────────────────────────┐  │
│  │  HandGestureController │  │  MoodyVoiceAssistant   │  │
│  │  (Background Thread)   │  │  (Background Thread)   │  │
│  │                       │  │                        │  │
│  │  MediaPipe Hands      │  │  speech_recognition    │  │
│  │  → pyautogui          │◄─┤  → CommandRegistry     │  │
│  │  (mouse/click/scroll) │  │  → pyttsx3 (TTS)      │  │
│  └───────────────────────┘  └────────────────────────┘  │
│                                                         │
│  ┌──────────────────┐  ┌─────────────────────────────┐  │
│  │  User Profiles    │  │  Advanced Analytics         │  │
│  │  (JSON files)     │  │  (Wellbeing, Productivity,  │  │
│  │                  │  │   Stability, PDF Reports)   │  │
│  └──────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### The connection flow:

1. **Launcher → Main App:** `common_launcher.py` handles login/signup, sets the `MOODY_USER` environment variable, then launches `fullemotionmodule.py` as a subprocess.

2. **Camera is shared:** One `cv2.VideoCapture(0)` instance is created. Both the emotion detection loop AND the hand gesture controller read frames from this same camera.

3. **Voice controls gestures:** The voice assistant has a `gesture_toggle_callback` — when you say "enable mouse" or "disable mouse", it calls back into the main app's `toggle_gesture_control()` method, turning gesture control on or off.

4. **Emotion → Actions:** When the detected emotion changes, the UI automatically updates the "Suggested Actions" panel on the right side. Each emotion has 10-17 specific action buttons (music, games, breathing exercises, journaling, etc.).

5. **Emotion → Analytics:** Every detected emotion is logged with a timestamp to the user's JSON file. The analytics engine tracks transitions, hourly patterns, streaks, and calculates wellbeing/productivity/stability scores.

6. **Background mode:** When you click "Background Run", the main window minimizes and a small draggable notification icon (🎭) appears on screen. Clicking it opens a compact popup with the current emotion and top suggested actions.

---

## 🖥️ UI Features Explained

### Main Window Layout (3-Column Design)

| Column | Content |
|--------|---------|
| **Left** | Live camera feed showing your face with emotion label overlaid. Below it: Start/Stop Detection, Enable Gestures, Background Run, and Enable Speech buttons. |
| **Middle** | Large emoji icon showing current emotion, emotion name, confidence percentage, and a recent history of detected emotions. |
| **Right** | Scrollable list of suggested action buttons based on your current emotion (e.g., play music, open games, breathing exercises, journaling). |

### Tabs (Notebook Interface)

| Tab | What It Shows |
|-----|-------------|
| **🎭 Emotion Recognition** | The main 3-column view described above. |
| **📈 Analytics** | Opens when you click "Analytics" button. Has sub-tabs: Today's Stats, This Week, Achievements, Advanced Analytics, Patterns & Insights. |
| **🎤 Voice Assistant** | Opens when you click "Enable Speech". Shows an activity log (chat-style), status indicator, start/stop buttons, background mode toggle, and a full command reference. |

### Emotion-Based Actions

Each emotion has a curated list of helpful actions. For example:

- **Happy** → Play "Happy" by Pharrell, Upbeat playlists, Pac-Man, Social media, Camera
- **Sad** → "Fix You" by Coldplay, Comedy videos, Self-care tips, Meditation, Helpline info
- **Angry** → "Weightless" by Marconi Union, Breathing exercises, Workout videos, Stress-relief games
- **Fear** → "Somewhere Over the Rainbow", Guided meditation, Emergency contacts, Grounding techniques
- **Neutral** → "Bohemian Rhapsody", 2048, Tetris, Learning resources, Productivity tools

### Background Mode

When running in background:
- A small **draggable icon** (🎭 with a notification badge) floats on the left side of the screen.
- Clicking it opens a **compact popup** showing the current emotion and top 6 action buttons.
- The popup is also draggable and remembers its position.
- Click **"Restore App"** in the popup to bring back the full window.

### User Profiles & Authentication

- **Login/Signup** is handled by the launcher (`common_launcher.py`).
- Passwords are hashed with SHA-256 before storage.
- Each user gets their own emotion log file and settings file.
- A "Guest" profile is available without a password.
- "Switch User" and "Logout" buttons redirect back to the launcher.

### Achievements & Notifications

- **Calm Mastery** — Triggered if you stay calm (no angry/fear spikes) for 2 hours.
- **Joy Spreader** — Triggered when you have 3+ happy moments in a session.
- Achievements appear as toast notifications at the top-right of the screen that auto-dismiss after 4 seconds.

---

## 🛠️ Tech Stack

| Technology | Used For |
|-----------|---------|
| **Python 3** | Core language |
| **Tkinter** | GUI framework (windows, buttons, tabs, canvas) |
| **OpenCV (cv2)** | Camera capture and image processing |
| **MediaPipe** | Face landmark detection (468 points) and hand landmark detection (21 points) |
| **scikit-learn / joblib** | Pre-trained emotion classification model |
| **pyautogui** | Mouse control, keyboard simulation, screenshots |
| **speech_recognition** | Microphone audio capture and Google Speech-to-Text |
| **pyttsx3** | Text-to-speech (offline, no API key needed) |
| **NumPy** | Numerical calculations for features and predictions |
| **Pillow (PIL)** | Image conversion for displaying camera frames in Tkinter |
| **pynput** | Keyboard simulation (used by voice assistant) |
| **reportlab** | PDF report generation (optional) |
| **pandas** | Data analysis for analytics (optional) |

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch through the dashboard:
   ```bash
   python launcher/common_launcher.py
   ```

3. Login or create an account, then the main emotion app opens automatically.

4. Click **"Start Detection"** to begin emotion recognition.

5. Click **"Enable Gestures"** to turn on hand gesture mouse control.

6. Click **"Enable Speech"** to open the voice assistant tab, then click **"Start Listening"** and say **"Hey Moody"**.
