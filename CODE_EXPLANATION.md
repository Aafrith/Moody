# Moody — Detailed Code Explanation

This document explains every section of **all code files** in the project in simple English so you can understand exactly what each part does and how it all works together.

**Files covered:**
- `fullemotionmodule.py` — The main app (3,411 lines)
- `hand_gesture.py` — Standalone gesture controller (253 lines)
- `voice_assistant.py` — Voice engine (1,555 lines)
- `live_emotion_inference.py` — Feature extraction (250+ lines)
- `advanced_analytics.py` — Analytics & report engine (400+ lines)
- `emotion_recognition_app.py` — Simplified emotion-only module
- `test_analytics.py` — Analytics test script
- `test_tabs.py` — Tab/analytics quick test
- `launcher/common_launcher.py` — Main launcher UI
- `launcher/theme_config.py` — Theme management
- `custom_commands.json` — Custom voice command definitions
- `requirements.txt` — All Python dependencies

---

# 📄 File 1: `fullemotionmodule.py` (3,411 lines — The Main App)

This is the heart of the entire project. It brings together emotion detection, hand gesture control, voice assistant, user profiles, analytics, background mode, and the full UI — all in one file.

---

## Imports (Lines 1–44)

```python
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import webbrowser
import subprocess
...
```

**What this does:** Loads all the libraries the app needs:
- `tkinter` — Python's built-in library for creating windows, buttons, labels, and the entire graphical interface.
- `cv2` (OpenCV) — Used to access the webcam and process video frames (read frames, flip them, convert colors, draw text on frames).
- `numpy` — For working with numbers and arrays (feature vectors, probability smoothing).
- `PIL` (Pillow) — Converts OpenCV images to a format that Tkinter can display.
- `threading` — Runs camera processing and gesture control in background threads so the UI doesn't freeze.
- `webbrowser` — Opens URLs in the user's default browser (for music, games, YouTube, etc.).
- `subprocess` — Opens desktop applications like Notepad, Paint, Calculator.
- `pyautogui` — Controls the mouse and keyboard programmatically (used by hand gestures).
- `json` — Reads and writes user profile and emotion log data files.
- `hashlib` — Hashes passwords with SHA-256 for secure storage.
- `joblib` — Loads the pre-trained machine learning model and label encoder from `.joblib` files.
- `mediapipe` — Google's library for detecting face landmarks (468 points) and hand landmarks (21 points).

It also imports custom modules:
- `theme_config` from the `launcher` folder — provides the current color theme.
- `AdvancedAnalytics` and `ReportGenerator` from `advanced_analytics.py` — the analytics engine.
- `MoodyVoiceAssistant` from `voice_assistant.py` — the voice control engine.
- `FEATURE_ORDER` and `compute_features` from `live_emotion_inference.py` — the 37 facial features and how to calculate them.

---

## Model Loading Setup (Lines 40–44)

```python
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model2")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.joblib")
LABELS_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
```

**What this does:** Defines the paths to the trained model files. The model directory is `model2/` inside the same folder. Two files are loaded:
- `emotion_model.joblib` — The actual machine learning model (trained classifier).
- `label_encoder.joblib` — Converts model output numbers (0, 1, 2...) to emotion names ("happy", "sad", etc.).

---

## HandGestureController Class (Lines 49–230)

This class handles everything about controlling the mouse using hand gestures.

### `__init__` (Constructor)

```python
class HandGestureController:
    def __init__(self):
        self.running = False
        self.is_active = False
        self.screen_width, self.screen_height = pyautogui.size()
        self.smooth_factor = 0.5
        self.prev_mouse_x, self.prev_mouse_y = 0, 0
        self.CLICK_HOLD_TIME = 0.5
        self.RIGHT_CLICK_HOLD = 0.3
        ...
```

**What this does:** Sets up all the variables the gesture controller needs:
- `running` — Whether the gesture system is running at all.
- `is_active` — Whether the virtual mouse is currently active (toggled by showing 5 fingers).
- `screen_width, screen_height` — Gets your screen resolution so hand positions can be mapped to screen coordinates.
- `smooth_factor = 0.5` — Smoothing value for cursor movement. Instead of jumping directly to where your finger points, it moves halfway (50%) between the old and new position each frame. This makes the cursor smooth and not jittery.
- `prev_mouse_x, prev_mouse_y` — Remembers the last mouse position for smoothing.
- `CLICK_HOLD_TIME = 0.5` — If you hold the click gesture for more than 0.5 seconds, it starts a drag instead of a click.
- `RIGHT_CLICK_HOLD = 0.3` — You need to hold the rock sign gesture for 0.3 seconds before it counts as a right click.
- `RIGHT_CLICK_COOLDOWN = 1.0` — After a right click, wait 1 second before allowing another one (prevents accidental double right-clicks).
- `SCROLL_INTERVAL = 0.15` — Scroll events fire every 0.15 seconds while the scroll gesture is active.
- `V_DEADZONE` — A "dead zone" around the screen center where scrolling doesn't happen. This prevents accidental scroll when your hand is roughly in the middle area.

### Helper Methods

```python
def landmarks_to_array(self, lm_list):
    return np.array([[lm.x, lm.y, lm.z] for lm in lm_list])
```

**What this does:** Converts MediaPipe's hand landmarks into a simple NumPy array. Each landmark has x (horizontal), y (vertical), and z (depth) values. The x and y values are normalized between 0 and 1 (0 = left/top of frame, 1 = right/bottom of frame).

```python
def finger_extended_np(self, lms, tip_idx, pip_idx):
    return lms[tip_idx, 1] < lms[pip_idx, 1]
```

**What this does:** Checks if a finger is extended (pointing up). It compares the y-coordinate of the fingertip with the y-coordinate of the middle joint (PIP). If the tip is above the PIP (lower y value because y increases downward), the finger is extended. Each finger has specific landmark indices:
- Index finger: tip = 8, PIP = 6
- Middle finger: tip = 12, PIP = 10
- Ring finger: tip = 16, PIP = 14
- Pinky: tip = 20, PIP = 18

```python
def thumb_really_extended_np(self, lms):
    tip = lms[self.mp_hands.HandLandmark.THUMB_TIP.value]
    ip = lms[self.mp_hands.HandLandmark.THUMB_IP.value]
    index_tip = lms[self.mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
    return tip[0] < ip[0] and abs(tip[0] - index_tip[0]) > 0.08
```

**What this does:** Special check for the thumb because it moves sideways, not up/down. It checks:
1. The thumb tip is to the left of the thumb's IP joint (`tip[0] < ip[0]` — in a mirrored/flipped frame, this means the thumb sticks out).
2. The thumb tip is at least 0.08 units away from the index finger — meaning the thumb is actually spread apart, not just resting beside the index finger.

```python
def five_fingers_extended(self, lms):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(finger_extended_np(lms, t, p) for t, p in zip(tips, pips)) and thumb_really_extended_np(lms)
```

**What this does:** Returns `True` only when ALL five fingers are extended (open hand). Used as the ON/OFF toggle for the virtual mouse.

### `start()` and `stop()`

```python
def start(self, cap):
    if not self.running:
        self.running = True
        self.cap = cap
        self.thread = threading.Thread(target=self._run_gesture_control, daemon=True)
        self.thread.start()
```

**What this does:** Starts gesture processing in a new background thread. It receives the camera capture object (`cap`) — this is the SAME camera that the emotion detection uses. The `daemon=True` means the thread will automatically stop when the main app closes.

### `_run_gesture_control()` — The Main Gesture Loop

This method runs continuously in the background thread. Each iteration:

1. **Reads a frame** from the shared camera.
2. **Flips it horizontally** (mirror effect so your left hand matches the left side of the screen).
3. **Converts to RGB** (MediaPipe expects RGB, but OpenCV captures in BGR).
4. **Processes with MediaPipe Hands** — detects hand landmarks.
5. **Checks for 5-finger toggle** — if all 5 fingers are extended and 1.5 seconds have passed since the last toggle, flip `is_active` ON or OFF.
6. **If active**, checks which fingers are extended and performs the appropriate action:

   **Cursor Movement:** If the index finger is extended, get its tip position, multiply by screen dimensions to get pixel coordinates, smooth with the previous position, and call `pyautogui.moveTo()`.

   **Left Click / Drag:** If thumb + index are extended (and others closed), start a timer. If released quickly → `pyautogui.click()`. If held > 0.5 seconds → `pyautogui.mouseDown()` starts dragging, and releasing later calls `pyautogui.mouseUp()`.

   **Right Click:** If index + pinky are extended (rock sign 🤘) and held for 0.3 seconds → `pyautogui.click(button="right")`. Has a 1-second cooldown.

   **Vertical Scroll:** If index + middle + ring are extended (3 fingers), measures the average y-position. If above screen center → scroll up. If below → scroll down. The deadzone prevents accidental scrolling when fingers are near the middle.

7. **Sleeps for 0.03 seconds** (~33 FPS) to avoid hogging CPU.

---

## EmotionRecognitionApp Class (Lines 234–3411)

This is the massive main class that creates the entire application.

### `__init__` (Constructor) (Lines 234–460)

**What this does:** Initializes everything when the app starts:

1. **Theme:** Loads colors from `theme_config` so the UI matches the user's selected theme.
2. **State variables:** Sets `current_emotion = "neutral"`, `detection_active = False`, and creates a `deque(maxlen=10)` for smoothing predictions.
3. **7 emotion labels:** `['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']`
4. **Hand gesture controller:** Creates a `HandGestureController()` instance.
5. **Popup/background mode variables:** For the floating icon and compact popup window.
6. **User profile paths:** Points to the `user_data/` directory.
7. **Analytics tracking:** Duration tracking per emotion, streak tracking, hourly patterns, transitions.
8. **Advanced analytics engine:** Creates an `AdvancedAnalytics()` instance.
9. **Emotion actions dictionary:** A huge mapping of emotion → list of action buttons. Each emotion has 13–17 actions, including specific songs (e.g., "Happy" by Pharrell for happy mood) and specific games (e.g., Pac-Man for happy, Slice Master for angry).

Then it calls:
- `self.setup_ui()` — Build the entire window layout.
- `self.setup_model()` — Load the ML model.
- `self.setup_camera()` — Initialize the webcam.
- `self.setup_responsive_layout()` — Handle window resizing.

Finally, it checks `MOODY_USER` environment variable (set by the launcher) to auto-login the user.

### `setup_ui()` — Building the Interface (Lines 462–700)

**What this does:** Creates the entire graphical interface using Tkinter widgets:

1. **Window setup:** Sets title, background color, minimum size (1000×700).
2. **Styles:** Configures dark theme styles for frames, labels, and buttons using `ttk.Style()`.
3. **Notebook (tabs):** Creates a `ttk.Notebook` widget — this is the tab container. The main tab is "🎭 Emotion Recognition". Analytics and Voice tabs are added dynamically when needed.
4. **Title bar:** Shows the app title and user controls (Switch User, Analytics, Download Report, Logout, Back to Dashboard buttons).
5. **Three columns** using grid layout:

   **Left Column — Camera:**
   - A `camera_container` frame that holds the video label.
   - `video_label` — Where camera frames are displayed.
   - Control buttons: Start Detection, Stop Detection, Enable Gestures, Background Run, Enable Speech.
   - A gesture status label ("Gesture Control: OFF/ON").

   **Middle Column — Emotion:**
   - A large emoji label (e.g., 😊) using 64pt font.
   - Emotion name label (e.g., "Happy").
   - Confidence percentage label (e.g., "Confidence: 87.3%").
   - Recent emotion history section.

   **Right Column — Actions:**
   - A scrollable `Canvas` with a `Scrollbar` containing action buttons.
   - The buttons change based on the current detected emotion.
   - `_on_canvas_configure` keeps the inner frame width synced with the canvas width.

### `setup_model()` (Lines ~750)

```python
def setup_model(self):
    self.model = joblib.load(MODEL_PATH)
    self.label_encoder = joblib.load(LABELS_PATH)
    self.model_loaded = True
```

**What this does:** Loads the pre-trained model and label encoder from the `model2/` folder using `joblib.load()`. If loading fails, it shows an error and sets `model_loaded = False`.

### `setup_camera()` (Lines ~760)

```python
def setup_camera(self):
    self.cap = cv2.VideoCapture(0)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    self.face_mesh = mp_face_mesh.FaceMesh(...)
```

**What this does:** Opens the default webcam (index 0) at 640×480 resolution and initializes MediaPipe FaceMesh with these settings:
- `static_image_mode=False` — Optimized for video (uses tracking between frames, not detection every frame).
- `max_num_faces=1` — Only track one face.
- `min_detection_confidence=0.5` — Face must be at least 50% confident to be detected.
- `min_tracking_confidence=0.5` — Tracking must be at least 50% confident to continue.

### `predict_emotion_from_frame()` — The Core ML Prediction (Lines ~790–840)

This is where the magic happens — every camera frame goes through this method to detect an emotion.

```python
def predict_emotion_from_frame(self, frame_bgr):
```

**Step-by-step what happens:**

1. **Convert to RGB:** `cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)` — MediaPipe expects RGB format.

2. **Run FaceMesh:** `self.face_mesh.process(rgb)` — Finds 468 face landmark points. If no face is found, returns `("neutral", 0.0)`.

3. **Extract landmarks:** Converts each landmark's normalized coordinates to pixel coordinates:
   ```python
   x_px = int(round(p.x * w))
   y_px = int(round(p.y * h))
   z_px = p.z * w
   ```

4. **Calculate 37 features:** `compute_features(landmarks, w, h)` — This calls into `live_emotion_inference.py` to compute all 37 facial measurements.

5. **Build feature vector:** Creates a NumPy array with features in the exact order the model was trained on (`FEATURE_ORDER`).

6. **Get model prediction:** Calls `model.predict_proba(x)` to get probabilities for each emotion. If that fails, falls back to `decision_function` with softmax.

7. **Smooth predictions:** Adds the raw probabilities to `self._proba_window` (a deque of last 10 predictions), then averages them all. This prevents the displayed emotion from flickering between frames.

8. **Get final label:** Takes the emotion with the highest smoothed probability, converts the number back to a name using `label_encoder.inverse_transform()`, then maps it to a standard label with `_canonical_label()`.

### `_canonical_label()` — Label Normalization

```python
def _canonical_label(self, label):
    mapping = {
        "anger": "angry", "angry": "angry",
        "happiness": "happy", "happy": "happy",
        "sadness": "sad", "sad": "sad",
        ...
    }
```

**What this does:** The model might output "anger" or "angry" or "sadness" or "sad". This method normalizes all variations to one of the 7 standard labels. If it encounters an unknown label, it defaults to "neutral".

### `detect_emotions()` — The Camera Loop (Lines ~850–900)

```python
def detect_emotions(self):
    while self.detection_active:
        ok, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        emotion, confidence = self.predict_emotion_from_frame(frame)
        ...
```

**What this does:** Runs in a background thread. It continuously:
1. Reads frames from the camera.
2. Flips horizontally (mirror effect).
3. Calls `predict_emotion_from_frame()` to get the emotion and confidence.
4. Overlays the emotion text on the frame.
5. Resizes the frame to fit the camera container while preserving aspect ratio.
6. Updates the UI using `root.after(0, ...)` — this is necessary because Tkinter can only be updated from the main thread, so `after(0, ...)` schedules the update on the main thread.
7. Sleeps 0.03 seconds for ~33 FPS.

### `update_emotion_display()` — Updating the UI (Lines ~910–960)

**What this does:** When a new emotion is detected:
1. Tracks how long the previous emotion lasted (for duration analytics).
2. Updates the emoji icon, text label, and confidence percentage.
3. If the emotion changed, updates the action suggestions panel and the background popup.
4. Logs the emotion with timestamp for analytics.
5. Checks for achievements (calm streak, happy spikes).

### `start_detection()` and `stop_detection()`

**`start_detection()`:** Checks that the model is loaded, camera is available, and user is logged in. Then sets `detection_active = True`, enables the gesture and background buttons, and starts the `detect_emotions()` loop in a new thread.

**`stop_detection()`:** Sets `detection_active = False` (the camera loop checks this flag and exits), saves the emotion log, and re-enables the start button.

### `toggle_gesture_control()` (Lines ~1000)

```python
def toggle_gesture_control(self):
    if not self.gesture_controller.running:
        self.gesture_controller.start(self.cap)
        # Show instructions messagebox
    else:
        self.gesture_controller.stop()
```

**What this does:** Toggles hand gesture mouse control. Passes the shared camera (`self.cap`) to the gesture controller. Shows a messagebox explaining all the gestures.

### Background Mode System (Lines ~1050–1200)

The background mode lets you minimize the main window while keeping emotion detection running.

**`enable_background_mode()`:** Minimizes the main window and shows a small floating notification icon.

**`show_notification_icon()`:** Creates a small 60×60 borderless `Toplevel` window with a purple circular button displaying 🎭. It has a red notification badge. It's:
- Always on top (`topmost=True`)
- Borderless (`overrideredirect=True`)
- Draggable (bind mouse press/motion events)
- Positioned on the left side of the screen

**`show_background_popup()`:** Creates a 320×420 borderless popup showing:
- Current emotion with emoji
- Top 6 suggested actions
- Gesture toggle button
- Restore App button
- It's draggable and remembers its last position

**`toggle_popup_from_notification()`:** When you click the notification icon, it either shows or hides the popup.

**`restore_from_background()`:** Brings back the main window, hides the popup and notification icon.

### Profile Management (Lines ~1350–1500)

**What this does:** Handles user accounts:

- `_load_profiles()` — Reads `profiles.json` from `user_data/` folder. This file stores `{username: password_hash}`.
- `_hash_password()` — Uses `hashlib.sha256()` to hash passwords (never stores plain text).
- `_verify_login()` — Checks if the given password's hash matches the stored hash. Guest accounts have no password.
- `_load_user_profile()` — Sets the current user, loads their settings file (`{username}_settings.json`) and emotion log (`{username}_emotions.json`).
- `_save_emotion_log()` — Writes the emotion log list to the user's JSON file.
- `_log_emotion()` — Adds each detected emotion with timestamp, confidence, and session ID to the log. Also feeds data into the analytics engine. Saves every 10 entries to avoid too many disk writes.
- `back_to_dashboard()` — Saves data, launches `common_launcher.py` as a new process, and closes the current window.

### Emotion Tracking & Achievements (Lines ~1500–1600)

**`_track_emotion_change()`:** Called whenever the emotion changes. Tracks:
- Emotion transitions (e.g., sad → happy).
- Happy spikes (happiness with >70% confidence).
- Emotion streaks (how long you stay in one emotion).
- Resets calm streak if angry/fear is detected with high confidence.

**`_check_achievements()`:** Runs every 5 minutes. Checks:
- **Calm Mastery:** If 2 hours have passed without angry/fear spikes → show toast notification.
- **Joy Spreader:** If you've had 3+ happy moments today → show toast notification.

**`_show_achievement()`:** Creates a toast-style popup at the top-right that auto-closes after 4 seconds.

### Analytics Panel (Lines ~1605–2750)

**`show_analytics_panel()`:** Creates an Analytics tab with 5 sub-tabs:

1. **Today's Stats** — Shows emotion distribution as progress bars, session duration, most common emotion.
2. **This Week** — Total mood checks, top 5 emotions with percentages, longest emotion streak.
3. **Achievements** — Calm streak progress, happy moment count, emotional balance (positive vs negative).
4. **Advanced Analytics** — Wellbeing score (0–100), productivity index (0–100), stress indicators (last 24h), emotional stability score, session statistics. Each score has a color-coded progress bar and interpretation text.
5. **Patterns & Insights** — Most common emotion transitions (e.g., "sad → happy: 12 times"), emotion patterns by hour of day, personalized insights, and recommendations.

### Voice Assistant Tab (Lines ~1730–1950)

**`show_voice_assistant_tab()`:** Creates a Voice Assistant tab with two columns:

**Left column — Activity Log:**
- A `tk.Text` widget styled like a chat log with colored text tags (user = blue, assistant = green, system = yellow, error = red).
- Shows timestamped messages.

**Right column — Control Panel:**
- Status display (Running/Stopped/Sleeping/Awake).
- Wake word indicator.
- Start/Stop buttons.
- Background mode checkbox.
- Wake word info ("Hey Moody").
- Full command reference organized by category (System Apps, Web, Volume, Gestures, Mouse, Keyboard, Scroll, System Control, Fun).

**`_start_voice_assistant()`:** Creates a `MoodyVoiceAssistant` instance with four callbacks:
- `on_status_change` → updates the status label.
- `on_log` → adds messages to the chat log.
- `on_wake` → updates the wake indicator.
- `gesture_toggle_callback` → lets voice commands enable/disable gesture mouse.

**`_voice_gesture_toggle(action)`:** Called when the voice assistant says "enable mouse" or "disable mouse". It calls the same `gesture_controller.start()` or `gesture_controller.stop()` methods as the button does, keeping everything in sync.

### Song & Game Methods (Lines ~2770–2950)

Each emotion has ~3 specific songs and ~2 specific games:

```python
def song_happy_pharrell(self):
    webbrowser.open("https://www.youtube.com/watch?v=ZbZSe6N_BXs")
    messagebox.showinfo("🎵 Now Playing", '"Happy" by Pharrell Williams')
```

**What this does:** Opens the YouTube video directly in the browser and shows a messagebox confirming what's playing. Each song and game is a separate method because each needs a different URL and different message.

### General Action Methods (Lines ~2950–3250)

These are the helper actions available across emotions — `play_calming_music()`, `open_meditation()`, `start_breathing_exercise()`, `open_journal()`, `show_emergency_contacts()`, etc. They all follow the same pattern: open a URL or launch an app, then show a messagebox confirmation.

### Report Generation (Lines ~3260–3340)

**`generate_report_dialog()`:** Shows a dialog with three export options:
- **PDF Report** — Uses `reportlab` library to create a professional PDF with scores, charts, and interpretation.
- **Excel Report** — Uses `pandas` to create a spreadsheet with multiple sheets.
- **JSON Export** — Raw data dump.

Each option opens a file save dialog, then calls the appropriate `ReportGenerator` method.

### Cleanup & Main Entry Point (Lines ~3350–3411)

**`__del__()`:** Releases the camera, closes MediaPipe, stops gesture controller, destroys popup windows.

**`on_closing()`:** Called when the user closes the window. Stops everything in order: detection, gestures, voice assistant, popup, notification icon, camera, face mesh. Then destroys the root window.

**`main()`:** Creates the Tkinter root window, creates the `EmotionRecognitionApp`, sets the close handler, and starts the event loop with `root.mainloop()`.

---

# 📄 File 2: `hand_gesture.py` (253 lines — Standalone Gesture Controller)

This is a standalone script that runs hand gesture mouse control independently, without the emotion detection UI. Useful for testing gestures on their own.

---

## CameraStream Class (Lines 13–41)

```python
class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.queue = Queue(maxsize=1)
        self.running = True
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
```

**What this does:** Creates a threaded camera reader. Instead of reading frames in the main loop (which would make it slower), this class:
1. Opens the webcam in a **background thread**.
2. Continuously reads frames and puts the latest one in a queue (size 1).
3. If a new frame arrives before the old one is consumed, the old frame is dropped.
4. The main loop calls `stream.read()` which gets the most recent frame instantly.

**Why threaded?** Without this, the main loop would block on `cap.read()` every frame, adding latency. With threaded capture, frames are always ready.

---

## Global Variables (Lines 46–78)

```python
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
smooth_factor = 0.5
is_active = False
...
```

**What this does:** Same variables as `HandGestureController` in the main module, but as global variables instead of class attributes. This is the standalone version so it uses simple global state.

---

## Utility Functions (Lines 80–100)

Same finger detection functions as the integrated version:
- `landmarks_to_array()` — Convert landmarks to numpy array.
- `finger_extended_np()` — Check if a finger is pointing up.
- `thumb_really_extended_np()` — Special thumb check.
- `five_fingers_extended()` — All fingers open check.

---

## Main Loop (Lines 105–253)

```python
stream = CameraStream()
with mp_hands.Hands(...) as hands:
    while True:
        frame = stream.read()
        frame = cv2.flip(frame, 1)
        ...
```

**What this does:** This is the main processing loop that runs forever:

1. **Read frame** from the threaded camera stream.
2. **Flip horizontally** for mirror effect.
3. **Convert to RGB** and process with MediaPipe Hands.
4. **Draw hand landmarks** on the frame using `mp_drawing.draw_landmarks()`.
5. **Check gestures** — Same logic as the integrated version:
   - 5 fingers → toggle ON/OFF
   - Index only → move cursor
   - Thumb + index → click/drag
   - Rock sign → right click
   - 3 fingers → scroll
6. **Display on-screen info:** Shows "ACTIVE" or "INACTIVE" status, current gesture name, and instructions.
7. **Show the frame** in an OpenCV window at ~30 FPS (rate-limited by `display_interval`).
8. **Exit** when 'q' key is pressed.

### Key difference from the integrated version
This standalone file shows a visible OpenCV window with the camera feed and gesture annotations. The integrated version (`fullemotionmodule.py`) does NOT show a separate window — it processes gestures silently in a background thread while the Tkinter UI displays the camera feed.

---

# 📄 File 3: `voice_assistant.py` (1,555 lines — Voice Engine)

This file contains the complete voice assistant system — wake word detection, speech recognition, text-to-speech, command registry with 100+ commands, and smart matching.

---

## Natural Language Processing (Lines 52–87)

```python
_NL_PREFIXES = [
    "can you", "could you", "would you", "will you",
    "please", "i want to", "i want you to", ...
]
_NL_SUFFIXES = ["please", "now", "for me", "right now", ...]

def _strip_natural_language(text):
```

**What this does:** People don't say commands robotically. They say things like "Can you please open YouTube for me now?" This function strips away the conversational filler words:
- Removes prefixes like "can you", "could you", "please", "I want to", "let's", "just"
- Removes suffixes like "please", "now", "for me", "right now"
- The result for the above example: "open youtube"

It loops repeatedly because there could be stacked prefixes ("can you please just...").

---

## Misheard Word Corrections (Lines 93–170)

```python
_MISHEARD = {
    "minimise": "minimize", "min eyes": "minimize",
    "screen short": "screenshot", "clothes": "close",
    "mause": "mouse", "school": "scroll",
    ...
}
```

**What this does:** Speech recognition often makes mistakes with certain words. This dictionary maps 70+ commonly misheard words to what the user actually meant. For example:
- "min eyes" → "minimize" (sounds similar)
- "screen short" → "screenshot"
- "clothes" → "close"
- "what's up" → "whatsapp"
- "school" → "scroll"

`_fix_misheard()` replaces these mistakes, checking longest matches first to avoid partial replacements.

---

## CommandRegistry Class (Lines 185–700)

This is the brain of the voice assistant — it holds every command the assistant can execute.

### Registration System

```python
def _add(self, trigger, handler, response, takes_query=False, aliases=None):
```

**What this does:** Each command is registered with:
- `trigger` — The primary phrase (e.g., "open notepad")
- `handler` — The function to run (e.g., `subprocess.Popen(["notepad.exe"])`)
- `response` — What Moody says back (e.g., "Opening Notepad")
- `takes_query` — Whether the command needs extra text (e.g., "search for **cats**")
- `aliases` — Alternative ways to say the same thing (e.g., "launch notepad", "start notepad", "run notepad")

### Commands by Category

**System Apps (~20 commands):**
- Open/close Notepad, Calculator, File Explorer, Task Manager, Settings, Control Panel, CMD, Paint, Snipping Tool, Word, Excel, PowerPoint, Photos, Camera, Clock, Maps, Store, Downloads, Desktop, Documents, Recycle Bin, Device Manager

**Websites (~30 commands):**
- Open Google, YouTube, Gmail, GitHub, Spotify, Netflix, Facebook, Instagram, Twitter, WhatsApp, LinkedIn, Reddit, Stack Overflow, ChatGPT, Amazon, eBay, Wikipedia, Twitch, Pinterest, TikTok, Google Drive, Google Docs, Google Sheets, Google Maps, Google Translate, Google Calendar, Outlook, Zoom, Discord, Telegram

**Search (4 commands):**
- Search Google for [query], Search YouTube for [query], Play on YouTube [query], Open website [URL]

**Volume & Media (10 commands):**
- Volume up/down/mute/max/min, Set volume to [0–100], Play/Pause, Next/Previous track, Stop music

**Screenshot (1 command):**
- Takes screenshot, saves to `~/Pictures/Moody_Screenshots/` with timestamp filename

**Window Management (8 commands):**
- Minimize, Maximize, Restore, Close window, Alt-Tab, Snap left/right, Show desktop

**Typing & Keyboard Shortcuts (~20 commands):**
- Type [text], Copy, Paste, Cut, Undo, Redo, Select All, Save, Save As, New Tab, Close Tab, Reopen Tab, Refresh, Zoom In/Out/Reset, Find, Find & Replace, Print, press Enter/Escape/Backspace/Delete/Tab/Space/Home/End/F5/F11

**Scrolling (8 commands):**
- Scroll up/down/left/right, Page up/down, Go to top/bottom

**Mouse & Gesture Control (10 commands):**
- Enable/Disable gesture mouse, Click, Double click, Right click, Move mouse up/down/left/right

**System Actions (10 commands):**
- Lock screen, Sleep, Brightness up/down, WiFi/Bluetooth/Display/Sound settings, Battery status

**Date & Time (4 commands):**
- Tell time, Tell date, Set timer, Set alarm

**Moody Commands (6 commands):**
- Go to sleep, Thank you, Hello, How are you, What can you do (help), Tell me a joke, Motivate me

**Close Apps (7 commands):**
- Close Chrome, Edge, Firefox, Word, Excel, PowerPoint, VLC

### Smart Matching System (Lines ~650–700)

```python
def match(self, text):
```

**What this does:** When the user says something, this method finds the best matching command through a multi-step process:

1. **Fix misheard words** — "Can you open the screen short tool" → "Can you open the screenshot tool"
2. **Strip natural language** — "Can you open the screenshot tool" → "open the screenshot tool"
3. **Exact match on triggers** — Checks all command triggers sorted by length (longest first), looking for exact matches or substring matches.
4. **Alias match** — Same process but against all registered aliases.
5. **Try again with raw text** — If stripping removed too much, try matching on the original.
6. **Fuzzy match** — Uses `SequenceMatcher` to find the closest command with at least 62% similarity. This catches misspellings and slight variations.

For commands that `takes_query=True` (like "search for"), the remaining text after the trigger is extracted as the query.

---

## Command Handler Methods (Lines 750–1200)

Each registered command points to a handler method. Some examples:

### App Launchers
```python
def _open_notepad(self):
    if platform.system() == "Windows":
        subprocess.Popen(["notepad.exe"])
    else:
        subprocess.Popen(["gedit"])
```
**What this does:** Opens the appropriate text editor based on the OS. Most app launchers follow this pattern — check the OS, then run the correct command.

### Volume Control
```python
def _volume_up(self):
    for _ in range(5):
        pyautogui.press('volumeup')
```
**What this does:** Presses the volume up key 5 times (each press is ~2% volume increase, so this is ~10% total).

### Set Volume
```python
def _set_volume(self, query=""):
    level = int(re.search(r'\d+', query).group())
    pyautogui.press('volumemute')  # start from known state
    pyautogui.press('volumemute')  # unmute
    for _ in range(level // 2):
        pyautogui.press('volumeup')
```
**What this does:** Extracts the number from the query (e.g., "50" from "set volume to 50"), then mutes/unmutes to reset volume, and presses volume up the appropriate number of times. Since each press is ~2%, it divides the level by 2.

### Gesture Mouse Control
```python
def _enable_gesture_mouse(self):
    cb = self.gesture_toggle_callback
    if cb:
        cb("enable")
```
**What this does:** Calls the callback function that was passed from the main app. This callback points to `_voice_gesture_toggle("enable")` in `fullemotionmodule.py`, which starts the gesture controller. This is how voice commands can control the hand gesture system.

### Screenshot
```python
def _take_screenshot(self):
    screenshot_dir = os.path.join(os.path.expanduser("~"), "Pictures", "Moody_Screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img = pyautogui.screenshot()
    img.save(filepath)
```
**What this does:** Takes a screenshot using pyautogui, saves it to `~/Pictures/Moody_Screenshots/` with a timestamped filename.

### Window Management
Uses `pyautogui.hotkey()` to press keyboard shortcuts:
- Minimize: `Win + Down` (twice)
- Maximize: `Win + Up`
- Close: `Alt + F4`
- Snap left: `Win + Left`
- Alt-Tab: `Alt + Tab`
- Show desktop: `Win + D`

### Help
```python
def _show_help(self):
    help_text = "I can do lots of things! ..."
    self.assistant.speak(help_text)
```
**What this does:** Speaks a long summary of everything the assistant can do.

### Jokes & Motivation
Has predefined lists of programmer jokes and motivational quotes, picks one randomly with `random.choice()`.

---

## MoodyVoiceAssistant Class (Lines ~1340–1555)

This is the core engine that handles the microphone and speech processing.

### `__init__` (Constructor)

```python
class MoodyVoiceAssistant:
    WAKE_WORDS = ["hey moody", "moody", "hay moody", "hey movie", ...]
```

**What this does:** Sets up:
- **Wake words** — 17 variations including common misheard versions ("hey movie", "hey modi", "hey buddy", "hey money").
- **State flags** — `running`, `listening`, `awake`, `background_mode`.
- **Speech recognizer** — `sr.Recognizer()` from the `speech_recognition` library. Sets:
  - `energy_threshold = 300` — Minimum audio energy level to consider as speech.
  - `dynamic_energy_threshold = True` — Automatically adjusts to ambient noise.
  - `pause_threshold = 0.8` — 0.8 seconds of silence = end of phrase.
- **TTS engine** — `pyttsx3.init()` for text-to-speech. Sets rate to 170 words/minute, volume to 90%, and tries to find a female voice (Zira on Windows).
- **Command registry** — Creates a `CommandRegistry` instance with a reference to itself and the gesture callback.
- **Awake timeout** — 45 seconds. If no command is heard for 45 seconds, the assistant goes back to sleep.

### `speak()` — Text-to-Speech

```python
def speak(self, text):
    self.on_log(f"🔊 Moody: {text}", "assistant")
    def _speak():
        with self._tts_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()
```

**What this does:** Converts text to speech and plays it from the speaker. Uses a lock (`_tts_lock`) so two messages don't try to speak at the same time. Runs in a separate thread so the listener isn't blocked while Moody is talking.

### `start()` and `stop()`

**`start()`:** Checks that `speech_recognition` is installed, sets all state flags, starts the listening thread.

**`stop()`:** Sets `running=False` which causes the listening loop to exit.

### `_listen_loop()` — Main Listening Loop

```python
def _listen_loop(self):
    self.microphone = sr.Microphone()
    with self.microphone as source:
        self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
    
    while self.running:
        self._listen_once()
        # Check awake timeout
        if self.awake and time.time() - self._last_command_time > 45:
            self.set_awake(False)
            self.speak("Going to sleep...")
```

**What this does:**
1. Opens the microphone and calibrates for ambient noise (takes 1.5 seconds).
2. Enters an infinite loop that repeatedly calls `_listen_once()`.
3. After each listen attempt, checks if the awake timeout has expired — if 45 seconds have passed since the last command, go back to sleep.

### `_listen_once()` — Single Listen Attempt

```python
def _listen_once(self):
    with sr.Microphone() as source:
        if self.awake:
            audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=12)
        else:
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
```

**What this does:**

**When sleeping (not awake):**
- Listens for up to 5 seconds with a 5-second phrase limit (short, because it's only looking for "Hey Moody").
- If it hears speech, sends it to Google Speech API.
- Checks if any wake word is in the recognized text.
- If found, sets `awake = True`, speaks "I'm listening!", and processes any remaining text after the wake word as a command.

**When awake:**
- Listens for up to 8 seconds with a 12-second phrase limit (longer, because commands can be complex).
- If the text starts with a wake word, strips it and processes the rest.
- If no wake word prefix, processes the entire text as a command.
- Updates `_last_command_time` to reset the 45-second timeout.

### `_process_command()` — Command Execution

```python
def _process_command(self, text):
    self.on_log(f"🗣️ You: {text}", "user")
    cmd_info, query = self.command_registry.match(text)
    
    if cmd_info:
        if cmd_info['response']:
            self.speak(cmd_info['response'])
        if cmd_info['takes_query']:
            cmd_info['handler'](query)
        else:
            cmd_info['handler']()
    else:
        self.speak("I'm not sure how to do that. Try saying 'help'.")
```

**What this does:**
1. Logs the user's command in the activity log.
2. Saves to command history.
3. Calls `command_registry.match(text)` to find the best matching command.
4. If a match is found:
   - Speaks the response (e.g., "Opening Notepad").
   - Calls the handler function, passing the query if needed.
5. If no match is found, tells the user it doesn't understand.

---

# 🔗 How the Three Files Connect

## Integration Points

### 1. Voice Assistant → Main App (via import)
```
fullemotionmodule.py imports:
    from voice_assistant import MoodyVoiceAssistant
```
The main app imports the `MoodyVoiceAssistant` class and creates an instance when the user clicks "Enable Speech".

### 2. Voice Assistant → Gesture Controller (via callback)
```
MoodyVoiceAssistant receives gesture_toggle_callback
    → calls callback("enable") or callback("disable")
    → which calls _voice_gesture_toggle() in fullemotionmodule.py
    → which calls gesture_controller.start() or stop()
```
When you say "enable mouse" or "disable mouse", the voice assistant calls a function pointer back into the main app, which toggles the gesture controller.

### 3. Camera Sharing
```
fullemotionmodule.py creates: self.cap = cv2.VideoCapture(0)
    → Emotion detection reads from self.cap in detect_emotions()
    → Hand gesture reads from same self.cap via gesture_controller.start(self.cap)
```
One camera is opened once and shared between emotion detection and gesture control. Both run in separate threads and read frames independently.

### 4. Feature Extraction (Live Emotion Inference)
```
fullemotionmodule.py imports:
    from live_emotion_inference import FEATURE_ORDER, compute_features
```
The `predict_emotion_from_frame()` method calls `compute_features()` from `live_emotion_inference.py` to calculate 37 facial features from the 468 MediaPipe landmarks.

### 5. Analytics Pipeline
```
fullemotionmodule.py imports:
    from advanced_analytics import AdvancedAnalytics, ReportGenerator
```
Every detected emotion is fed into `AdvancedAnalytics` which scores wellbeing, productivity, and stability in real-time. `ReportGenerator` is used only when the user clicks "Download Report".

### 6. Theme Sharing
```
launcher/theme_config.py writes: current_theme.json
    → fullemotionmodule.py reads it via get_theme_colors()
    → emotion_recognition_app.py reads it the same way
```
The launcher saves the user's chosen theme, and all modules read from the same `current_theme.json` file so colors are consistent everywhere.

### 7. Standalone vs Integrated
- `hand_gesture.py` — Works completely on its own. Opens its own camera, shows its own OpenCV window. No emotion detection, no UI.
- `voice_assistant.py` — Has the `MoodyVoiceAssistant` class that can be imported. Can also be used standalone if needed.
- `emotion_recognition_app.py` — A minimal, self-contained emotion viewer with no gesture/voice/analytics. Useful for testing the core detection pipeline.
- `fullemotionmodule.py` — The full app that imports and integrates everything.
- `live_emotion_inference.py` — Feature extraction library; imported but never run directly.
- `advanced_analytics.py` — Analytics library; imported but never run directly.

---

## Summary: What Each File Is For

| File | Purpose | Lines | Standalone? |
|------|---------|-------|-------------|
| `fullemotionmodule.py` | **Main app** — Everything together: emotion detection, hand gestures, voice assistant, UI, profiles, analytics | 3,411 | Yes (main entry point) |
| `hand_gesture.py` | **Standalone gesture controller** — Test hand gestures without the emotion UI | 253 | Yes (run directly) |
| `voice_assistant.py` | **Voice engine** — Wake word detection, 100+ commands, speech recognition, smart matching | 1,555 | Imported by main app |
| `live_emotion_inference.py` | **Feature extractor** — Calculates 37 facial measurements from MediaPipe landmarks | 250+ | Imported (library) |
| `advanced_analytics.py` | **Analytics + report engine** — Wellbeing/productivity scores, PDF/Excel/JSON export | 400+ | Imported (library) |
| `emotion_recognition_app.py` | **Minimal emotion viewer** — Core detection only, no gestures/voice/analytics | ~200 | Yes (run directly) |
| `test_analytics.py` | **Analytics test** — Verifies all analytics calculations and report generation | ~120 | Yes (run directly) |
| `test_tabs.py` | **Tab test** — Quick check that analytics tabs populate with correct data | ~70 | Yes (run directly) |
| `launcher/common_launcher.py` | **Dashboard launcher** — The purple Moody home screen with user login and module launch | 500+ | Yes (main entry point) |
| `launcher/theme_config.py` | **Theme manager** — Dark/light color palettes, reads/writes `current_theme.json` | ~100 | Imported (library) |
| `custom_commands.json` | **Custom voice commands** — User-defined extra commands for the voice assistant | JSON | Data file |
| `requirements.txt` | **Dependencies list** — All Python packages the project needs | — | Config file |

---

# 📄 File 4: `live_emotion_inference.py` (250+ lines — Feature Extractor)

This file is a **library** — it is never run on its own. It is imported by `fullemotionmodule.py` and `emotion_recognition_app.py` to calculate the 37 facial measurements used by the ML model.

---

## Why Is This a Separate File?

The feature extraction logic is the same whether you're running the full app or the minimal app. By putting it in its own file, both modules can import it without duplicating code. If you ever need to change how a feature is calculated (e.g., improve smile detection), you only change it in one place.

---

## `FEATURE_ORDER` — The 37 Feature Names (Lines 16–35)

```python
FEATURE_ORDER = [
    'mouth_movement', 'mouth_aspect_ratio', 'lip_corner_distance', 'jaw_drop',
    'left_eye_movement', 'right_eye_movement', 'left_eyebrow_movement', 'right_eyebrow_movement',
    'left_eyebrow_slope', 'right_eyebrow_slope', 'eyebrow_asymmetry', 'nostril_flare',
    ...
]
```

**What this does:** This list defines the EXACT order that features must be passed to the ML model. The model was trained with features in this specific order — if you mix them up, you get wrong predictions. This constant is the "contract" between the feature extractor and the model.

---

## Geometry Helper Functions (Lines 40–85)

These are small math utilities used by `compute_features()`:

```python
def sdist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])
```
**`sdist(a, b)`** — Calculates the straight-line (Euclidean) distance between two 2D points. Used for measuring distances like "how far apart are the lip corners?".

```python
def sratio(num, den):
    if num is None or den in (None, 0):
        return 0.0
    return float(num) / float(den)
```
**`sratio(num, den)`** — Safe division that returns 0.0 instead of crashing on division by zero. Most features are ratios (e.g., mouth height ÷ mouth width) so safe division is essential.

```python
def angle_deg(a, b, c):
```
**`angle_deg(a, b, c)`** — Calculates the angle at point `b` formed by points `a`, `b`, `c`. Used for jaw angle detection.

```python
def point_line_signed_distance(p, a, b):
```
**`point_line_signed_distance(p, a, b)`** — Measures how far point `p` is from the line drawn through `a` and `b`, with sign (positive = one side, negative = other side). Used for detecting smile curvature — a positive value means the mouth center is ABOVE the line between lip corners (smile), negative means it's below (frown).

---

## Head Pose Estimation (Lines 88–125)

```python
def estimate_head_pose(landmarks, w, h):
    idxs = [1, 152, 33, 263, 61, 291]
    ...
    ok, rvec, tvec = cv2.solvePnP(pts3d, pts2d, cam_mtx, dist, ...)
    R, _ = cv2.Rodrigues(rvec)
    return yaw, pitch, roll
```

**What this does:** Uses OpenCV's `solvePnP` (solve Perspective-n-Point) to estimate the 3D orientation of the head from 6 key landmark points:
- Nose tip (index 1)
- Chin (index 152)
- Left eye outer corner (index 33)
- Right eye outer corner (index 263)
- Left mouth corner (index 61)
- Right mouth corner (index 291)

It matches these 2D points on the camera image against a generic 3D face model (standard measurements in millimetres), then solves for the rotation that would project the 3D model onto those 2D points. This gives three rotation angles:
- **Yaw** — Left/right head rotation (shaking head "no")
- **Pitch** — Up/down head tilt (nodding "yes")
- **Roll** — Side-to-side head tilt (like leaning your head on your shoulder)

These head pose angles help distinguish, for example, a fearful downward glance (high pitch) from a surprised upward look.

---

## `compute_features()` — The 37 Measurements (Lines 127–230)

```python
def compute_features(landmarks, w, h):
    L = lambda i: safe_L(landmarks, i)
    total_reference = sdist(L(4), L(6))
    ...
```

**What this does:** Takes the 468 face landmark points from MediaPipe and calculates all 37 features. Everything is computed as a **ratio** rather than a raw pixel value. This is crucial — if you're far from the camera your face is small (small pixel values) and if you're close it's large (large pixel values), but ratios stay consistent.

**The reference distance** (`total_reference`) is the distance between landmarks 4 and 6 (two points on the nose bridge). All other distances are divided by this to normalize for face size.

**Key features computed:**

| Feature | What it measures | Landmarks used |
|---------|-----------------|----------------|
| `mouth_aspect_ratio` | How open the mouth is (height ÷ width) | 13 (top lip), 14 (bottom lip), 61 (left corner), 291 (right corner) |
| `lip_corner_distance` | How wide the mouth is relative to face | 61, 291 |
| `jaw_drop` | How far the chin has dropped | 152 (chin), 14 (bottom lip) |
| `left_eye_ear` | Eye Aspect Ratio — how open the left eye is | 159 (top), 145 (bottom), 33 (left), 133 (right) |
| `left_eyebrow_slope` | Whether the eyebrow is angled up or down | 65 (inner brow), 159 (eye center) |
| `eyebrow_asymmetry` | Whether one eyebrow is higher than the other | Both eyebrow heights |
| `smile_intensity` | How much the mouth corners curve upward | 61, 291, 13, 14 |
| `mouth_curvature` | Distance of mouth center from lip corner line | 61, 291, mouth midpoint |
| `pose_yaw / pitch / roll` | Head orientation angles | 6 key points via solvePnP |
| `jaw_angle_deg` | The angle at the chin | 234 (jaw left), 152 (chin), 454 (jaw right) |
| `nostril_flare` | How wide the nostrils are | 98 (left nostril), 327 (right nostril) |

All 37 values are packed into the order defined by `FEATURE_ORDER` and returned as a dictionary.

---

# 📄 File 5: `advanced_analytics.py` (400+ lines — Analytics & Report Engine)

This file provides two classes: `AdvancedAnalytics` (computes scores) and `ReportGenerator` (exports data to files). Both are imported and used by the main app.

---

## `AdvancedAnalytics` Class

### `__init__` (Constructor)

```python
class AdvancedAnalytics:
    def __init__(self):
        self.emotion_transitions = defaultdict(lambda: defaultdict(int))
        self.hourly_emotions = defaultdict(lambda: defaultdict(int))
        self.stress_indicators = []
        self.productivity_score = 0.0
        self.wellbeing_score = 0.0
```

**What this does:** Sets up three data structures:
- `emotion_transitions` — A nested dictionary. E.g., `transitions["happy"]["sad"] = 5` means "the emotion changed from happy to sad 5 times". Used for the "Patterns & Insights" tab.
- `hourly_emotions` — Counts what emotions occurred at each hour of the day. E.g., `hourly["09"]["happy"] = 3`. Used for the hourly patterns chart.
- `stress_indicators` — A list of high-confidence negative emotions (angry/fear/sad > 70% confidence) with timestamps. Used for the stress count display.

### Score Calculations

**`calculate_wellbeing_score(emotion_log)`:**

Looks at the last 50 emotion log entries. Each emotion has a "happiness weight":
- Happy = 100, Surprise = 80, Neutral = 60, Disgust = 40, Sad = 30, Fear = 20, Angry = 10

Multiplies each weight by the detection confidence, sums them up, and divides by total confidence. Result is 0–100. High = positive emotional state.

**`calculate_productivity_score(emotion_log)`:**

Looks at the last 30 entries. Splits emotions into two groups:
- Productive: `{happy, neutral, surprise}`
- Distracting: `{angry, sad, fear, disgust}`

Score = (productive count ÷ total count) × 100. Simple but effective — if 70% of recent emotions are productive ones, you score 70.

**`calculate_stability_score(emotion_log)`:**

Looks at the last 30 entries and counts how many times the emotion changed from one entry to the next. The "change rate" is changes ÷ total entries. Stability = 100 − (change_rate × 100). Low fluctuation = high stability.

### Event Tracking

**`track_emotion_transition(from_emotion, to_emotion)`:** Increments `emotion_transitions[from][to]` by 1. Called every time the detected emotion changes.

**`track_hourly_emotion(emotion)`:** Gets the current hour (0–23) and increments `hourly_emotions[hour][emotion]`. Called every time an emotion is logged.

**`add_stress_indicator(emotion, confidence)`:** If the emotion is angry/fear/sad AND the confidence is above 70%, records it with a timestamp in `stress_indicators`. Used to identify high-intensity negative moments.

**`get_recent_stress_count(hours=24)`:** Filters `stress_indicators` to only count events within the last N hours. Returns the count as an integer.

### Insights Generator

**`generate_insights(emotion_log, wellbeing_score, productivity_score)`:**

Produces a list of personalized text messages based on the data:
- Most common emotion → "Your dominant emotion recently is happy (12 occurrences)"
- Wellbeing ≥ 75 → positive message; < 50 → suggests self-care
- Productivity ≥ 70 → positive message; < 40 → suggests breaks
- Stress count > 10 → recommends meditation
- Stability < 40 → recommends calming routine; > 70 → positive message

### Color & Interpretation Helpers

**`get_score_color(score)`:** Returns a hex color string — green (`#00ff88`) for ≥ 75, orange (`#ffaa00`) for ≥ 50, red (`#ff4444`) below 50. Used to color-code the progress bars in the analytics tab.

**`get_wellbeing_interpretation(score)`:** Returns a one-line human-readable description:
- ≥ 80 → "Excellent! You're in a great emotional state."
- ≥ 60 → "Good! Overall positive wellbeing."
- ≥ 40 → "Fair. Consider some self-care activities."
- < 40 → "Needs attention. Please take care of yourself."

---

## `ReportGenerator` Class

All methods are `@staticmethod` — no instance needed, just call them directly.

### `generate_pdf_report(filename, user, emotion_log, analytics)`

**What this does:** Creates a multi-page PDF file using the `reportlab` library:
1. **Title page** — "🎭 Emotion Analytics Report", user name, date, total entries logged.
2. **Executive Summary** — Wellbeing, productivity, and stability scores with interpretation text.
3. **Emotion Distribution table** — Lists each emotion with count and percentage.
4. **Personalized Insights** — Bullet points from `generate_insights()`.
5. **Stress Analysis** — Stress event count for the last 24 hours with Low/Moderate/High classification.
6. **Page break**, then **Recent Emotion Log** — A table showing the last 20 detected emotions with timestamps and confidence values.

The PDF uses a purple color scheme (`#6a4c93`) for headings to match the Moody brand.

### `generate_excel_report(filename, user, emotion_log, analytics)`

**What this does:** Creates an Excel `.xlsx` file using `pandas`:
- **Sheet 1: Emotion Log** — Every emotion entry with timestamp, emotion name, and confidence.
- **Sheet 2: Summary** — Wellbeing, productivity, and stability scores with metric name, score, and interpretation.
- **Sheet 3: Emotion Distribution** — Count and percentage for each emotion.
- **Sheet 4: Insights** — One insight per row.

Uses `pandas.ExcelWriter` with the `openpyxl` engine for writing. If `openpyxl` is not installed, it falls back to a CSV file instead.

### `generate_json_report(filename, user, emotion_log, analytics)`

**What this does:** Exports everything as raw JSON — the simplest format. Includes:
- Username, report generation timestamp
- All three scores (wellbeing, productivity, stability)
- Every entry from `emotion_log`
- All stress indicators
- All insights

This is the most flexible format — other programs can import it.

---

# 📄 File 6: `emotion_recognition_app.py` (Simplified Emotion-Only Module)

This is a **minimal version** of the emotion detection system — it only does real-time emotion recognition with a basic UI. No voice assistant, no hand gestures, no analytics, no user profiles.

---

## When to Use This vs `fullemotionmodule.py`

| Feature | `emotion_recognition_app.py` | `fullemotionmodule.py` |
|---------|------------------------------|------------------------|
| Emotion detection | ✅ | ✅ |
| Suggested actions | ✅ (simplified) | ✅ (13–17 buttons) |
| Hand gesture control | ❌ | ✅ |
| Voice assistant | ❌ | ✅ |
| User profiles & login | ❌ | ✅ |
| Analytics & reports | ❌ | ✅ |
| Background mode | ❌ | ✅ |

**Use this file** when you want to quickly test the emotion detection pipeline, or if you want a stripped-down version for a lower-power machine.

---

## `EmotionRecognitionApp` Class

Its structure mirrors `fullemotionmodule.py` exactly for the detection parts:
- Same `__init__` with `current_emotion`, `detection_active`, `_proba_window` deque
- Same 7 emotion labels: `['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']`
- Same `emotion_actions` dictionary (slightly shorter action lists)
- Same `setup_model()`, `setup_camera()`, `predict_emotion_from_frame()`, `detect_emotions()` methods calling `compute_features()` from `live_emotion_inference.py`

The UI is simpler — there's no analytics tab, no profile panel, no gesture or speech buttons.

---

# 📄 File 7: `test_analytics.py` (Analytics Test Script)

This is a **developer test script** — you run it from the terminal to verify that the analytics engine is working correctly before running the full app.

---

## What It Tests (9 Tests)

```
Test 1: Import Check       — Can AdvancedAnalytics and ReportGenerator be imported?
Test 2: Dependencies       — Are reportlab and pandas installed?
Test 3: Initialization     — Can AdvancedAnalytics() be created without errors?
Test 4: Sample Data        — Do the three score calculations return sensible numbers?
Test 5: Transitions        — Does track_emotion_transition() record correctly?
Test 6: Hourly Patterns    — Does track_hourly_emotion() record correctly?
Test 7: Stress Indicators  — Does add_stress_indicator() and get_recent_stress_count() work?
Test 8: Insights           — Does generate_insights() return non-empty insights?
Test 9: JSON Report        — Does generate_json_report() create a valid file?
```

### How to Run

```
cd emotion_gesture
python test_analytics.py
```

Each test prints ✓ for pass or ✗ with the error message for fail. If all pass, it's safe to run the full app.

---

# 📄 File 8: `test_tabs.py` (Tab快速 Test Script)

This is an even simpler test script focused specifically on verifying that the data fed into the **Advanced Analytics** and **Patterns & Insights** tabs is correct.

---

## What It Does

Creates a small set of 6 sample emotion entries, runs them through all four analytics calculations, and prints the results alongside what the Analytics tab should display. It also:
- Populates `emotion_transitions` with 4 sample transitions
- Tracks hourly patterns for 6 sample emotions
- Verifies that `generate_insights()` returns insights

**This is useful when:** You changed something in `advanced_analytics.py` and want to quickly verify the numbers look right before opening the full UI.

---

# 📄 Launcher Files

---

## `launcher/common_launcher.py` — The Main Dashboard (500+ lines)

This is the first thing users see when they launch Moody. It's the purple home screen that lets users log in and choose which module to start.

---

### Color Constants (Lines 1–18)

```python
BG_TOP = "#7C6EE6"
BG_BOTTOM = "#5B2C83"
BTN_PINK = "#E879F9"
BTN_RED = "#F43F5E"
BTN_BLUE = "#60A5FA"
```

**What this does:** Defines the purple/violet color scheme used throughout the launcher. Having them as named constants means you can change the whole color scheme by editing just one line per color.

---

### `draw_rounded_rect()` & `create_rounded_button()`

```python
def draw_rounded_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    points = [x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, ...]
    return canvas.create_polygon(points, smooth=True, **kwargs)
```

**What this does:** Tkinter doesn't have a built-in "rounded rectangle" shape. These functions create one by drawing a polygon with 12 corner points and using `smooth=True` to round the edges. `create_rounded_button()` places this shape on a `Canvas` widget and adds hover effects (color changes when you mouse over it) and click handling.

---

### `GradientFrame` Class

```python
class GradientFrame(tk.Canvas):
    def _draw_gradient(self, event=None):
        for i in range(h):
            nr = int(r1 + (r2 - r1) * i / h)
            ...
```

**What this does:** A `Canvas` that draws a vertical color gradient (top color → bottom color). It works by drawing one horizontal line per pixel of height, each line a slightly different color blended between the two endpoints. The gradient re-draws itself when the window is resized (via the `<Configure>` event).

---

### Welcome Messages & Tips

```python
WELCOME_GREETINGS = ["Welcome back! Ready to explore?", ...]
TIPS = ["💡 Tip: Use the Emotion module to control your PC with facial expressions.", ...]
FEATURES = [("🎭", "Emotion Detection", "Real-time facial expression analysis ..."), ...]
```

**What this does:** Pre-written strings displayed randomly in the launcher. `WELCOME_GREETINGS` shows a random greeting each time you open the launcher. `TIPS` shows usage tips that cycle. `FEATURES` are the three feature cards shown in the main body.

---

### `MoodyLauncher` Class

This is the main launcher window, subclassing `tk.Tk` directly (it IS the root window).

**`__init__`:**
- Sets window title to "MOODY", geometry to 900×750, minimum size 360×520 (works on tablets/phones).
- Detects the virtual environment's Python executable (`venv/Scripts/python.exe`) so sub-processes use the right Python.
- Sets DPI awareness on Windows so the UI isn't blurry on high-DPI screens.
- Tracks `_is_mobile` flag for responsive layout, `_tip_index` for cycling tips.

**Responsive layout:** Listens to `<Configure>` events (window resize). When the width drops below 620 pixels (`MOBILE_BREAKPOINT`), it switches to a mobile layout — hides some elements, makes buttons full-width, etc.

**User authentication:**
- Reads `user_data/profiles.json` to get all user accounts.
- Shows a login dialog with username/password fields.
- Hashes the typed password with SHA-256 and compares to the stored hash.
- On success, sets the `MOODY_USER` environment variable before launching sub-processes — this is how the main emotion app knows who's logged in.

**Module launching:**
- When you click "Launch Emotion Module", it runs `python emotion_gesture/fullemotionmodule.py` as a subprocess.
- The `MOODY_USER` env variable passes the username into the launched module.
- The launcher either stays open or closes depending on settings.

---

## `launcher/theme_config.py` — Theme Manager (~100 lines)

This file manages the dark/light color theme across the entire application.

---

### `THEMES` Dictionary

```python
THEMES = {
    "dark": {
        "bg_primary": "#1a1a1a",
        "bg_secondary": "#2a2a2a",
        "accent_primary": "#238636",
        "accent_emotion": "#9d4edd",
        ...
    },
    "light": {
        "bg_primary": "#ffffff",
        "bg_secondary": "#f6f8fa",
        "accent_primary": "#2da44e",
        "accent_emotion": "#8250df",
        ...
    }
}
```

**What this does:** Defines two complete color palettes (dark and light). Each palette has 15 color keys:
- `bg_primary/secondary/tertiary` — Background colors (3 shades for layering)
- `bg_hover` — Background color when you hover over an element
- `accent_primary` — Green, used for "start" / positive buttons
- `accent_secondary` — Blue, used for secondary buttons
- `accent_emotion` — Purple, used for emotion-related UI elements
- `accent_speech` — Teal, used for voice assistant elements
- `accent_danger` — Red, used for stop/error buttons
- `accent_warning` — Orange, used for warnings
- `text_primary/secondary/muted` — Three levels of text brightness
- `border` — Border/divider color

### `get_current_theme()` / `set_current_theme()`

**What this does:** Reads and writes `current_theme.json` in the same directory. The file stores just `{"theme": "dark"}` or `{"theme": "light"}`. If the file doesn't exist or is corrupted, it defaults to `"dark"`.

### `get_theme_colors(theme_name=None)`

**What this does:** Returns the full color dictionary for the requested theme (or current theme if none specified). Called by every module on startup to load their colors.

### `toggle_theme()`

**What this does:** Reads the current theme, flips it to the other one, saves it, and returns the new theme name. Used by UI buttons that switch between dark and light mode.

---

# 📄 Data & Config Files

---

## `custom_commands.json` — Custom Voice Commands

```json
{
  "open project folder": {
    "type": "run",
    "windows": "explorer C:\\Projects",
    "mac": "open ~/Projects",
    "linux": "xdg-open ~/Projects"
  },
  "start meeting": {
    "type": "url",
    "url": "https://meet.google.com"
  },
  "start coding": {
    "type": "run",
    "windows": "code",
    "mac": "open -a 'Visual Studio Code'",
    "linux": "code"
  }
}
```

**What this does:** Lets users add their own voice commands without touching any Python code. Each key is what you say out loud. The `type` field can be:
- `"run"` — Launch a shell command. Specify different commands for `windows`/`mac`/`linux`.
- `"url"` — Open a URL in the default browser.

These are loaded by the voice assistant at startup and registered alongside the built-in commands.

---

## `user_data/profiles.json` — User Accounts

```json
{
    "Guest": "",
    "Dilexan": "sha256_hash_here",
    "Lahiru": "sha256_hash_here"
}
```

**What this does:** Stores all registered usernames and their hashed passwords. The key is the username, the value is the SHA-256 hash of their password. Guest users have an empty string (no password required). **Plain text passwords are never stored.**

---

## `user_data/{username}_emotions.json` — Emotion History

```json
[
    {"emotion": "happy", "confidence": 0.87, "timestamp": "2026-03-08T14:32:11", "session_id": "abc123"},
    {"emotion": "neutral", "confidence": 0.75, "timestamp": "2026-03-08T14:32:14", "session_id": "abc123"}
]
```

**What this does:** A running list of every detected emotion for that user. Each entry has four fields:
- `emotion` — One of the 7 standard labels.
- `confidence` — How confident the model was (0.0 to 1.0).
- `timestamp` — When it was detected (ISO 8601 format).
- `session_id` — A unique ID for the detection session, so you can tell which entries came from the same session.

The file is saved every 10 entries during detection (to avoid excessive disk writes) and again when detection is stopped.

---

## `user_data/{username}_settings.json` — User Settings

Stores per-user preferences such as the default theme, whether background mode is enabled, and notification preferences. Loaded on login and saved when the user logs out or changes settings.

---

## `launcher/current_theme.json` — Active Theme

```json
{"theme": "dark"}
```

**What this does:** A tiny file storing just the current theme name. Written by `theme_config.set_current_theme()` and read by every module on startup. This is how the theme persists across app restarts.

---

# 📦 `requirements.txt` — Dependencies

This file lists every Python package that Moody needs. You install them all at once with:

```
pip install -r requirements.txt
```

---

## Dependency Groups

### Computer Vision & Image Processing
| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | 4.8.1.78 | Read webcam, flip/resize frames, draw text, `solvePnP` for head pose |
| `mediapipe` | 0.10.13 | Detect 468 face landmarks and 21 hand landmarks per frame |
| `Pillow` | 10.1.0 | Convert OpenCV BGR images to Tkinter-compatible format |

### Machine Learning & Data Processing
| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 2.0.0 | Numerical arrays for feature vectors, probability averaging |
| `joblib` | ≥ 1.4.0 | Load the pre-trained `.joblib` model files |
| `scikit-learn` | ≥ 1.7.0 | The ML classifier itself (used inside the loaded model) |
| `pandas` | ≥ 2.2.0 | Excel report generation (multi-sheet `.xlsx` files) |

### Speech Recognition & Text-to-Speech
| Package | Version | Purpose |
|---------|---------|---------|
| `SpeechRecognition` | 3.10.0 | Converts microphone audio to text via Google Speech API |
| `pyttsx3` | 2.90 | Converts text back to speech (text-to-speech engine) |
| `PyAudio` | 0.2.14 | Low-level microphone access (required by SpeechRecognition) |
| `vosk` | ≥ 0.3.45 | Offline speech recognition (fallback if no internet) |

### System Automation & Control
| Package | Version | Purpose |
|---------|---------|---------|
| `pyautogui` | 0.9.54 | Move mouse, click, scroll, press keyboard keys |
| `PyGetWindow` | 0.0.9 | Get window titles and positions for gesture-based window management |
| `pynput` | 1.7.6 | Low-level keyboard and mouse input monitoring |
| `psutil` | 5.9.8 | Get battery status, system info for voice commands |

### OCR (Optional)
| Package | Version | Purpose |
|---------|---------|---------|
| `pytesseract` | 0.3.10 | Optical Character Recognition for reading text on screen |

### Report Generation
| Package | Version | Purpose |
|---------|---------|---------|
| `reportlab` | 4.0.7 | Generate professional PDF reports |
| `openpyxl` | 3.1.2 | Read/write Excel `.xlsx` files (used by pandas) |

### Additional Utilities
| Package | Version | Purpose |
|---------|---------|---------|
| `setuptools` | ≥ 75.0.0 | Provides the `distutils` module removed in Python 3.12+ |
| `pywin32` | 306 | Windows-specific system APIs (Windows only) |

---

## ⚠️ Known Issue Fix: `No module named 'distutils'`

**Problem:** When running the Voice Assistant tab on Python 3.12 or newer, this error appears:
```
❌ Microphone error: No module named 'distutils'
```

**Why it happens:** Python 3.12 removed the built-in `distutils` module from the standard library. Some older packages (like PyAudio) still import `distutils` internally.

**Fix:** Install `setuptools`, which re-provides the `distutils` module:
```
pip install setuptools
```

Or install all requirements at once (now includes `setuptools`):
```
pip install -r requirements.txt
```

`setuptools>=75.0.0` has been added to `requirements.txt` so that new installations automatically include it.

---