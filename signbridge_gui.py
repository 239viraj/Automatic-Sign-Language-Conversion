
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import customtkinter as ctk
import threading
import time

from PIL import Image
from collections import deque, Counter
import pandas as pd


# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_PATH = "sign_model.pkl"

STABLE_FRAMES = 15
PREDICTION_HISTORY_SIZE = 15

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540


# =========================================================
# 2. LOAD MODEL
# =========================================================

try:
    model = joblib.load(MODEL_PATH)

    if getattr(model, "n_features_in_", None) != 126:
        raise ValueError(
            "The model must accept 126 features. "
            "Please use your newly trained two-hand model."
        )

    print("SignBridge AI model loaded successfully!")
    print("Features:", model.n_features_in_)
    print("Signs:", len(model.classes_))

except Exception as error:
    raise SystemExit(
        f"Could not load the correct model: {error}"
    )


# =========================================================
# 3. INITIALIZE MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# =========================================================
# 4. APPLICATION VARIABLES
# =========================================================

camera = None
camera_running = False
closing = False

sentence = []

prediction_history = deque(
    maxlen=PREDICTION_HISTORY_SIZE
)

last_sign = None
stable_sign = None
same_sign_count = 0
last_added_sign = None

current_prediction = "Waiting for camera..."
detected_hands = 0

# Latest webcam frame is shared with a separate facial-analysis worker.
latest_frame = None
latest_frame_lock = threading.Lock()
face_analysis_running = False
face_analysis_busy = False
face_analysis_lock = threading.Lock()

face_age_text = "Age estimate: Waiting..."
face_expression_text = "Expression: Waiting..."
face_gender_text = "Gender prediction: Waiting..."

camera_lock = threading.Lock()
speech_lock = threading.Lock()


# =========================================================
# 5. TEXT TO SPEECH
# =========================================================

def speak_text(text):
    """Speak the recognized sentence in a background thread."""

    if not text.strip():
        return

    def speech_worker():
        try:
            with speech_lock:
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.setProperty("volume", 1.0)

                engine.say(text.replace("_", " "))
                engine.runAndWait()
                engine.stop()

        except Exception as error:
            print("Speech error:", error)

    threading.Thread(
        target=speech_worker,
        daemon=True
    ).start()


# =========================================================
# 6. EXTRACT 126 FEATURES
# =========================================================

def extract_hand_features(results, frame):
    """
    Extract left-hand landmarks followed by right-hand
    landmarks.

    Each hand contributes 21 landmarks x 3 coordinates
    = 63 features.

    Total = 126 features.
    """

    left_features = np.zeros(63, dtype=np.float32)
    right_features = np.zeros(63, dtype=np.float32)

    hand_count = 0

    if not results.multi_hand_landmarks:
        return np.concatenate(
            [left_features, right_features]
        ), 0

    for index, hand_landmarks in enumerate(
        results.multi_hand_landmarks
    ):

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # MediaPipe handedness must match the convention
        # used by collect_data.py.
        hand_label = None

        if (
            results.multi_handedness
            and index < len(results.multi_handedness)
        ):
            hand_label = (
                results.multi_handedness[index]
                .classification[0].label
            )

        # Frames are mirrored using cv2.flip(frame, 1).
        # Swap labels to get anatomical left/right hands.
        if hand_label == "Left":
            hand_label = "Right"
        elif hand_label == "Right":
            hand_label = "Left"

        coordinates = []

        for landmark in hand_landmarks.landmark:
            coordinates.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        coordinates = np.array(
            coordinates,
            dtype=np.float32
        )

        if hand_label == "Left":
            left_features = coordinates

        elif hand_label == "Right":
            right_features = coordinates

        else:
            # Do not guess the hand identity if MediaPipe
            # does not provide a valid handedness label.
            continue

        hand_count += 1

    features = np.concatenate([
        left_features,
        right_features
    ])

    return features, hand_count


# =========================================================
# 7. FACIAL ANALYSIS (DeepFace, runs in background)
# =========================================================

def update_face_labels(age, expression, gender):
    """Update the facial-analysis panel on the Tkinter main thread."""
    if closing:
        return
    age_label.configure(text=f"Approximate age: {age}")
    expression_label.configure(text=f"Expression: {expression}")
    gender_label.configure(text=f"Gender prediction: {gender}")


def face_analysis_worker():
    """
    Analyze a recent webcam frame roughly every two seconds.
    DeepFace is imported here so a missing/broken installation
    does not prevent the sign-recognition GUI from opening.
    """
    global face_analysis_running
    global face_analysis_busy
    global face_age_text, face_expression_text, face_gender_text

    try:
        from deepface import DeepFace
    except Exception as error:
        print("DeepFace could not be loaded:", error)
        if not closing:
            app.after(
                0,
                update_face_labels,
                "Unavailable",
                "DeepFace unavailable",
                "Unavailable"
            )
        face_analysis_running = False
        return

    while face_analysis_running and not closing:
        with latest_frame_lock:
            frame_copy = latest_frame.copy() if latest_frame is not None else None

        if frame_copy is not None:
            try:
                # DeepFace expects a BGR image, which is what OpenCV provides.
                result = DeepFace.analyze(
                    img_path=frame_copy,
                    actions=["age", "gender", "emotion"],
                    enforce_detection=False,
                    silent=True
                )

                if isinstance(result, list):
                    result = result[0]

                age_value = result.get("age", "Unknown")
                if isinstance(age_value, (int, float)):
                    age_start = max(0, (int(float(age_value)) // 5) * 5)
                    age_text = f"{age_start}-{age_start + 4} years"
                else:
                    age_text = str(age_value)

                emotion_values = result.get("emotion", {})
                expression = (
                    max(emotion_values, key=emotion_values.get).title()
                    if isinstance(emotion_values, dict) and emotion_values
                    else "Unknown"
                )

                gender_values = result.get("gender", {})
                gender = (
                    max(gender_values, key=gender_values.get).title()
                    if isinstance(gender_values, dict) and gender_values
                    else str(gender_values or "Unknown")
                )

                face_age_text = age_text
                face_expression_text = expression
                face_gender_text = gender

                if not closing:
                    app.after(
                        0,
                        update_face_labels,
                        age_text,
                        expression,
                        gender
                    )

            except Exception as error:
                print("Facial analysis error:", error)
                if not closing:
                    app.after(
                        0,
                        update_face_labels,
                        "Not detected",
                        "Not detected",
                        "Not detected"
                    )

        # Do not analyze every frame; DeepFace can be computationally expensive.
        for _ in range(20):
            if not face_analysis_running or closing:
                break
            time.sleep(0.1)

    face_analysis_running = False


# =========================================================
# 8. RESET RECOGNITION STATE
# =========================================================

def reset_recognition():
    global last_sign
    global stable_sign
    global same_sign_count
    global last_added_sign

    prediction_history.clear()

    last_sign = None
    stable_sign = None
    same_sign_count = 0
    last_added_sign = None


# =========================================================
# 8. ADD RECOGNIZED SIGN
# =========================================================

def update_prediction(prediction):

    global last_sign
    global stable_sign
    global same_sign_count
    global last_added_sign

    if prediction is None:
        reset_recognition()
        return

    prediction_history.append(prediction)

    most_common = Counter(
        prediction_history
    ).most_common(1)[0][0]

    if most_common == last_sign:
        same_sign_count += 1
    else:
        last_sign = most_common
        same_sign_count = 1
        stable_sign = None

    if same_sign_count >= STABLE_FRAMES:

        stable_sign = most_common

        if stable_sign != last_added_sign:

            sentence.append(stable_sign)
            last_added_sign = stable_sign

            update_sentence_display()

            print("Recognized:", stable_sign)


# =========================================================
# 9. SENTENCE DISPLAY
# =========================================================

def update_sentence_display():
    sentence_text = " ".join(sentence).replace(
        "_", " "
    )

    app.after(
        0,
        lambda text=sentence_text:
        sentence_label.configure(
            text=text if text else "Your sentence appears here..."
        )
    )


# =========================================================
# 10. CAMERA WORKER
# =========================================================

def camera_worker():

    global camera
    global camera_running
    global current_prediction
    global detected_hands
    global latest_frame

    while camera_running and not closing:

        with camera_lock:
            cap = camera

        if cap is None:
            break

        success, frame = cap.read()

        if not success:
            app.after(
                0,
                lambda: status_label.configure(
                    text="Camera error",
                    text_color="#EF4444"
                )
            )
            break

        # Mirror the webcam image.
        frame = cv2.flip(frame, 1)

        frame = cv2.resize(
            frame,
            (CAMERA_WIDTH, CAMERA_HEIGHT)
        )

        with latest_frame_lock:
            latest_frame = frame.copy()

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands_detector.process(rgb_frame)

        prediction = None

        if results.multi_hand_landmarks:

            features, hand_count = extract_hand_features(
                results,
                frame
            )

            detected_hands = hand_count

            if hand_count > 0:

                # Preserve the feature names used during training when available.
                if hasattr(model, "feature_names_in_"):
                    feature_row = pd.DataFrame(
                        [features],
                        columns=model.feature_names_in_
                    )
                    prediction_array = model.predict(feature_row)
                else:
                    prediction_array = model.predict(
                        features.reshape(1, -1)
                    )

                prediction = str(
                    prediction_array[0]
                )

                update_prediction(prediction)

            else:
                reset_recognition()

        else:
            detected_hands = 0
            reset_recognition()

        current_prediction = (
            prediction if prediction
            else "No hand detected"
        )

        # Display current prediction on the camera.
        cv2.putText(
            frame,
            f"Sign: {current_prediction}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 100),
            2
        )

        cv2.putText(
            frame,
            f"Hands: {detected_hands}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Convert the frame for Tkinter.
        display_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(display_frame)

        image = image.resize(
            (640, 360),
            Image.Resampling.LANCZOS
        )

        photo = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(640, 360)
        )

        app.after(
            0,
            update_camera_image,
            photo,
            current_prediction,
            detected_hands
        )

        time.sleep(0.01)

    with camera_lock:
        cap = camera
        camera = None

    if cap is not None:
        cap.release()

    camera_running = False

    if not closing:
        app.after(0, camera_stopped_ui)


# =========================================================
# 11. UPDATE CAMERA DISPLAY
# =========================================================

def update_camera_image(
    photo,
    prediction,
    hand_count
):

    if closing:
        return

    camera_label.configure(image=photo, text="")
    camera_label.image = photo

    current_sign_label.configure(
        text=prediction.replace("_", " ")
    )

    hands_label.configure(
        text=f"Hands detected: {hand_count}"
    )


# =========================================================
# 12. START CAMERA
# =========================================================

def start_camera():

    global camera
    global camera_running

    if camera_running:
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        cap.release()

        status_label.configure(
            text="Unable to open camera",
            text_color="#EF4444"
        )

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    with camera_lock:
        camera = cap

    camera_running = True

    status_label.configure(
        text="Camera running",
        text_color="#22C55E"
    )

    start_button.configure(
        state="disabled"
    )

    stop_button.configure(
        state="normal"
    )

    threading.Thread(
        target=camera_worker,
        daemon=True
    ).start()

    global face_analysis_running
    if not face_analysis_running:
        face_analysis_running = True
        threading.Thread(
            target=face_analysis_worker,
            daemon=True
        ).start()


# =========================================================
# 13. STOP CAMERA
# =========================================================

def stop_camera():

    global camera_running

    camera_running = False

    status_label.configure(
        text="Stopping camera...",
        text_color="#F59E0B"
    )

    stop_button.configure(
        state="disabled"
    )


def camera_stopped_ui():

    start_button.configure(
        state="normal"
    )

    stop_button.configure(
        state="disabled"
    )

    status_label.configure(
        text="Camera stopped",
        text_color="#F59E0B"
    )

    camera_label.configure(
        image=None,
        text="Camera stopped\n\nClick START CAMERA to begin",
        text_color="#9CA3AF"
    )

    camera_label.image = None


# =========================================================
# 14. SPEAK SENTENCE
# =========================================================

def speak_sentence():

    if not sentence:
        status_label.configure(
            text="No sentence to speak",
            text_color="#F59E0B"
        )
        return

    text = " ".join(sentence)

    speak_text(text)

    status_label.configure(
        text="Speaking sentence...",
        text_color="#22C55E"
    )


# =========================================================
# 15. CLEAR SENTENCE
# =========================================================

def clear_sentence():

    sentence.clear()

    reset_recognition()

    update_sentence_display()

    status_label.configure(
        text="Sentence cleared",
        text_color="#60A5FA"
    )


# =========================================================
# 16. CLOSE APPLICATION
# =========================================================

def close_application():

    global closing
    global camera_running
    global face_analysis_running

    closing = True
    camera_running = False
    face_analysis_running = False

    with camera_lock:
        cap = camera

    if cap is not None:
        cap.release()

    hands_detector.close()

    app.destroy()


# =========================================================
# 17. CREATE GUI
# =========================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()

app.title("SignBridge AI | Sign Language Recognition")

app.geometry("1250x850")
app.minsize(1000, 700)

app.configure(
    fg_color="#0B1120"
)

app.protocol(
    "WM_DELETE_WINDOW",
    close_application
)


# =========================================================
# 18. MAIN HEADER
# =========================================================

header = ctk.CTkFrame(
    app,
    fg_color="#111827",
    corner_radius=15
)

header.pack(
    fill="x",
    padx=20,
    pady=(20, 10)
)

title_label = ctk.CTkLabel(
    header,
    text="SignBridge AI",
    font=ctk.CTkFont(
        family="Segoe UI",
        size=30,
        weight="bold"
    ),
    text_color="#60A5FA"
)

title_label.pack(
    side="left",
    padx=25,
    pady=18
)

subtitle_label = ctk.CTkLabel(
    header,
    text="AI-Powered Sign Language Recognition",
    font=ctk.CTkFont(
        size=14
    ),
    text_color="#D1D5DB"
)

subtitle_label.pack(
    side="left",
    padx=10
)

status_label = ctk.CTkLabel(
    header,
    text="Camera stopped",
    font=ctk.CTkFont(
        size=14,
        weight="bold"
    ),
    text_color="#F59E0B"
)

status_label.pack(
    side="right",
    padx=25
)


# =========================================================
# 19. MAIN CONTENT
# =========================================================

main_frame = ctk.CTkFrame(
    app,
    fg_color="transparent"
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=10
)

main_frame.grid_columnconfigure(
    0,
    weight=3
)

main_frame.grid_columnconfigure(
    1,
    weight=2
)

main_frame.grid_rowconfigure(
    0,
    weight=1
)


# =========================================================
# 20. CAMERA PANEL
# =========================================================

camera_panel = ctk.CTkFrame(
    main_frame,
    fg_color="#111827",
    corner_radius=15
)

camera_panel.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 10),
    pady=0
)

camera_heading = ctk.CTkLabel(
    camera_panel,
    text="LIVE CAMERA",
    font=ctk.CTkFont(
        size=18,
        weight="bold"
    ),
    text_color="#E5E7EB"
)

camera_heading.pack(
    anchor="w",
    padx=20,
    pady=(20, 12)
)

camera_label = ctk.CTkLabel(
    camera_panel,
    text="Camera is currently off\n\nClick START CAMERA to begin",
    font=ctk.CTkFont(size=16),
    text_color="#9CA3AF",
    fg_color="#030712",
    corner_radius=10,
    width=640,
    height=360
)

camera_label.pack(
    padx=15,
    pady=10,
    fill="both",
    expand=True
)

current_sign_label = ctk.CTkLabel(
    camera_panel,
    text="Waiting for sign...",
    font=ctk.CTkFont(
        size=22,
        weight="bold"
    ),
    text_color="#34D399"
)

current_sign_label.pack(
    pady=(10, 5)
)

hands_label = ctk.CTkLabel(
    camera_panel,
    text="Hands detected: 0",
    font=ctk.CTkFont(size=14),
    text_color="#D1D5DB"
)

hands_label.pack(
    pady=(0, 15)
)


# =========================================================
# 21. RIGHT SIDE PANEL
# =========================================================

right_panel = ctk.CTkFrame(
    main_frame,
    fg_color="transparent"
)

right_panel.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(10, 0)
)


# ---------------------------------------------------------
# SENTENCE CARD
# ---------------------------------------------------------

sentence_card = ctk.CTkFrame(
    right_panel,
    fg_color="#111827",
    corner_radius=15
)

sentence_card.pack(
    fill="x",
    pady=(0, 15)
)

ctk.CTkLabel(
    sentence_card,
    text="RECOGNIZED SENTENCE",
    font=ctk.CTkFont(
        size=17,
        weight="bold"
    ),
    text_color="#60A5FA"
).pack(
    anchor="w",
    padx=20,
    pady=(20, 12)
)

sentence_label = ctk.CTkLabel(
    sentence_card,
    text="Your sentence appears here...",
    font=ctk.CTkFont(size=16),
    text_color="#F9FAFB",
    wraplength=350,
    justify="left",
    anchor="w"
)

sentence_label.pack(
    fill="x",
    padx=20,
    pady=(0, 25)
)


# ---------------------------------------------------------
# MODEL INFORMATION CARD
# ---------------------------------------------------------

model_card = ctk.CTkFrame(
    right_panel,
    fg_color="#111827",
    corner_radius=15
)

model_card.pack(
    fill="x",
    pady=(0, 15)
)

ctk.CTkLabel(
    model_card,
    text="MODEL INFORMATION",
    font=ctk.CTkFont(
        size=17,
        weight="bold"
    ),
    text_color="#60A5FA"
).pack(
    anchor="w",
    padx=20,
    pady=(20, 15)
)

for text in [
    f"Recognized signs: {len(model.classes_)}",
    f"Landmark features: {model.n_features_in_}",
    "Hand detection: Up to 2 hands",
    "Recognition: Random Forest"
]:

    ctk.CTkLabel(
        model_card,
        text="• " + text,
        font=ctk.CTkFont(size=14),
        text_color="#D1D5DB"
    ).pack(
        anchor="w",
        padx=22,
        pady=5
    )

ctk.CTkLabel(
    model_card,
    text=""
).pack(pady=3)


# ---------------------------------------------------------
# FACIAL ANALYSIS CARD
# ---------------------------------------------------------

face_card = ctk.CTkFrame(
    right_panel,
    fg_color="#111827",
    corner_radius=15
)

face_card.pack(
    fill="x",
    pady=(0, 15)
)

ctk.CTkLabel(
    face_card,
    text="FACIAL ANALYSIS (ESTIMATES)",
    font=ctk.CTkFont(size=16, weight="bold"),
    text_color="#A78BFA"
).pack(
    anchor="w",
    padx=20,
    pady=(15, 10)
)

age_label = ctk.CTkLabel(
    face_card,
    text="Approximate age: Waiting...",
    font=ctk.CTkFont(size=13),
    text_color="#D1D5DB"
)
age_label.pack(anchor="w", padx=20, pady=3)

expression_label = ctk.CTkLabel(
    face_card,
    text="Expression: Waiting...",
    font=ctk.CTkFont(size=13),
    text_color="#D1D5DB"
)
expression_label.pack(anchor="w", padx=20, pady=3)

gender_label = ctk.CTkLabel(
    face_card,
    text="Gender prediction: Waiting...",
    font=ctk.CTkFont(size=13),
    text_color="#D1D5DB"
)
gender_label.pack(anchor="w", padx=20, pady=(3, 15))


# ---------------------------------------------------------
# CONTROLS CARD
# ---------------------------------------------------------

controls_card = ctk.CTkScrollableFrame(
    right_panel,
    fg_color="#111827",
    corner_radius=15
)

controls_card.pack(
    fill="both",
    expand=True
)

ctk.CTkLabel(
    controls_card,
    text="CONTROLS",
    font=ctk.CTkFont(
        size=17,
        weight="bold"
    ),
    text_color="#60A5FA"
).pack(
    anchor="w",
    padx=20,
    pady=(20, 15)
)


start_button = ctk.CTkButton(
    controls_card,
    text="▶  START CAMERA",
    command=start_camera,
    height=42,
    corner_radius=10,
    fg_color="#16A34A",
    hover_color="#15803D",
    font=ctk.CTkFont(
        size=14,
        weight="bold"
    )
)

start_button.pack(
    fill="x",
    padx=20,
    pady=7
)


stop_button = ctk.CTkButton(
    controls_card,
    text="■  STOP CAMERA",
    command=stop_camera,
    height=42,
    corner_radius=10,
    fg_color="#DC2626",
    hover_color="#B91C1C",
    font=ctk.CTkFont(
        size=14,
        weight="bold"
    ),
    state="disabled"
)

stop_button.pack(
    fill="x",
    padx=20,
    pady=7
)


speak_button = ctk.CTkButton(
    controls_card,
    text="🔊  SPEAK SENTENCE",
    command=speak_sentence,
    height=42,
    corner_radius=10,
    fg_color="#2563EB",
    hover_color="#1D4ED8",
    font=ctk.CTkFont(
        size=14,
        weight="bold"
    )
)

speak_button.pack(
    fill="x",
    padx=20,
    pady=7
)


clear_button = ctk.CTkButton(
    controls_card,
    text="✕  CLEAR SENTENCE",
    command=clear_sentence,
    height=42,
    corner_radius=10,
    fg_color="#4B5563",
    hover_color="#374151",
    font=ctk.CTkFont(
        size=14,
        weight="bold"
    )
)

clear_button.pack(
    fill="x",
    padx=20,
    pady=7
)

quit_button = ctk.CTkButton(
    controls_card,
    text="✕  QUIT APPLICATION",
    command=close_application,
    height=42,
    corner_radius=10,
    fg_color="#7F1D1D",
    hover_color="#991B1B",
    font=ctk.CTkFont(
        size=14,
        weight="bold"
    )
)

quit_button.pack(
    fill="x",
    padx=20,
    pady=7
)


# =========================================================
# 22. FOOTER
# =========================================================

footer = ctk.CTkLabel(
    app,
    text=(
        "SignBridge AI  •  30 Sign Classes  •  "
        "Two-Hand Recognition  •  Text-to-Speech"
    ),
    font=ctk.CTkFont(size=12),
    text_color="#9CA3AF"
)

footer.pack(
    pady=(0, 15)
)


# =========================================================
# 23. START APPLICATION
# =========================================================

app.mainloop()

