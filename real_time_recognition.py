
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import pyttsx3
import threading
import time

from deepface import DeepFace
from collections import deque, Counter


# ==================================================
# 1. LOAD TRAINED SIGN LANGUAGE MODEL
# ==================================================

MODEL_FILE = "sign_model.pkl"
EXPECTED_FEATURES = 126

model = joblib.load(MODEL_FILE)

if model.n_features_in_ != EXPECTED_FEATURES:
    raise ValueError(
        f"Your model expects {model.n_features_in_} features, "
        f"but this application requires {EXPECTED_FEATURES}.\n"
        "Update collect_data.py and train_model.py, then retrain the model."
    )

print("Two-hand sign language model loaded successfully!")


# ==================================================
# 2. INITIALIZE TEXT-TO-SPEECH
# ==================================================

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)


# ==================================================
# 3. INITIALIZE MEDIAPIPE HANDS
# ==================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ==================================================
# 4. START WEBCAM
# ==================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    hands.close()
    raise SystemExit("Cannot open camera.")


# ==================================================
# 5. SIGN RECOGNITION VARIABLES
# ==================================================

sentence = []

last_sign = None
same_sign_count = 0

STABLE_FRAMES = 15
prediction_history = deque(maxlen=15)

current_prediction = None


# ==================================================
# 6. FACIAL ANALYSIS VARIABLES
# ==================================================

face_age = "Waiting..."
age_range = "Waiting..."
face_emotion = "Waiting..."
gender_presentation = "Waiting..."

face_lock = threading.Lock()
worker_lock = threading.Lock()

analysis_running = False
ANALYSIS_INTERVAL = 2.0
last_analysis_time = 0


# ==================================================
# 7. ESTIMATE AGE RANGE
# ==================================================

def get_age_range(age):

    if age <= 12:
        return "0-12"

    elif age <= 19:
        return "13-19"

    elif age <= 29:
        return "20-29"

    elif age <= 39:
        return "30-39"

    elif age <= 59:
        return "40-59"

    return "60+"


# ==================================================
# 8. BACKGROUND FACIAL ANALYSIS
# ==================================================

def analyze_face(frame):

    global face_age
    global age_range
    global face_emotion
    global gender_presentation
    global analysis_running

    try:

        results = DeepFace.analyze(
            img_path=frame,
            actions=["age", "emotion", "gender"],
            detector_backend="opencv",
            enforce_detection=True,
            silent=True
        )

        if results:

            face = results[0]

            age = int(round(face["age"]))
            estimated_range = get_age_range(age)
            emotion = face["dominant_emotion"]

            gender_scores = face.get("gender", {})

            if gender_scores:
                predicted_gender = max(
                    gender_scores,
                    key=gender_scores.get
                )
            else:
                predicted_gender = "Unavailable"

            with face_lock:

                face_age = str(age)
                age_range = estimated_range
                face_emotion = str(emotion)
                gender_presentation = str(predicted_gender)

        else:

            with face_lock:
                face_age = "Unavailable"
                age_range = "No face detected"
                face_emotion = "Unavailable"
                gender_presentation = "Unavailable"

    except Exception as error:
        print("Facial analysis error:", error)

    finally:

        with worker_lock:
            analysis_running = False


# ==================================================
# 9. EXTRACT BOTH HANDS IN A CONSISTENT ORDER
# ==================================================

def extract_hand_features(results):

    # 21 landmarks × 3 coordinates = 63 features per hand.
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    detected_hands = 0

    if not results.multi_hand_landmarks:
        return None, 0

    handedness_results = results.multi_handedness or []

    for index, hand_landmarks in enumerate(
        results.multi_hand_landmarks
    ):

        # Draw detected hand.
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Extract 63 landmark features.
        hand_features = []

        for landmark in hand_landmarks.landmark:
            hand_features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # MediaPipe handedness labels are based on its expected
        # input orientation. This application mirrors the camera
        # frame before processing, so account for that convention.
        label = None

        if index < len(handedness_results):
            classification = (
                handedness_results[index].classification
            )

            if classification:
                label = classification[0].label

        # Because the frame is mirrored before processing,
        # swap the handedness label to obtain the user's
        # anatomical left/right hand.
        if label == "Left":
            label = "Right"
        elif label == "Right":
            label = "Left"

        if label == "Left":
            left_hand = hand_features

        elif label == "Right":
            right_hand = hand_features

        else:
            # Do not guess which anatomical hand was detected.
            # An unknown hand is not assigned a fabricated side.
            continue

        detected_hands += 1

        # Show the detected hand label.
        wrist = hand_landmarks.landmark[0]

        height, width = frame.shape[:2]

        wrist_x = int(wrist.x * width)
        wrist_y = int(wrist.y * height)

        cv2.putText(
            frame,
            label + " hand",
            (wrist_x, max(25, wrist_y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    if detected_hands == 0:
        return None, 0

    # IMPORTANT:
    # The collector must save features in exactly this order:
    # left hand (63) followed by right hand (63).
    features = left_hand + right_hand

    return np.asarray(features, dtype=np.float32).reshape(1, -1), detected_hands


# ==================================================
# 10. MAIN APPLICATION
# ==================================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Cannot read camera.")
            break

        # Mirror webcam.
        frame = cv2.flip(frame, 1)

        current_time = time.time()

        # ------------------------------------------
        # AUTOMATIC FACIAL ANALYSIS SCHEDULER
        # ------------------------------------------

        should_start_analysis = False

        with worker_lock:

            if (
                not analysis_running
                and current_time - last_analysis_time
                >= ANALYSIS_INTERVAL
            ):
                analysis_running = True
                last_analysis_time = current_time
                should_start_analysis = True

        if should_start_analysis:

            analysis_thread = threading.Thread(
                target=analyze_face,
                args=(frame.copy(),),
                daemon=True
            )

            analysis_thread.start()

        # ------------------------------------------
        # HAND DETECTION
        # ------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        features, hand_count = extract_hand_features(results)

        current_prediction = None

        # ------------------------------------------
        # SIGN PREDICTION
        # ------------------------------------------

        if features is not None:

            feature_names = getattr(
                model,
                "feature_names_in_",
                None
            )

            if feature_names is not None:

                feature_input = pd.DataFrame(
                    features,
                    columns=feature_names
                )

                prediction = model.predict(feature_input)

            else:

                prediction = model.predict(features)

            current_prediction = str(prediction[0])

            prediction_history.append(current_prediction)

            most_common_sign = Counter(
                prediction_history
            ).most_common(1)[0][0]

            if most_common_sign == last_sign:
                same_sign_count += 1

            else:
                last_sign = most_common_sign
                same_sign_count = 1

            # Add a sign once it remains stable.
            if same_sign_count == STABLE_FRAMES:

                if (
                    not sentence
                    or sentence[-1] != most_common_sign
                ):
                    sentence.append(most_common_sign)

                    print("Word added:", most_common_sign)

                # Prevent adding the same sign every frame.
                same_sign_count += 1

        else:

            last_sign = None
            same_sign_count = 0
            prediction_history.clear()

        # ------------------------------------------
        # DISPLAY CURRENT SIGN
        # ------------------------------------------

        if current_prediction is not None:

            current_text = (
                f"Current: {current_prediction} "
                f"({hand_count} hand(s))"
            )

        else:
            current_text = "Current: No hand detected"

        cv2.putText(
            frame,
            current_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # ------------------------------------------
        # DISPLAY SENTENCE
        # ------------------------------------------

        sentence_text = " ".join(sentence)

        if len(sentence_text) > 45:
            sentence_text = "..." + sentence_text[-42:]

        cv2.putText(
            frame,
            "Sentence: " + sentence_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ------------------------------------------
        # DISPLAY FACIAL ANALYSIS
        # ------------------------------------------

        with face_lock:
            displayed_age_range = age_range
            displayed_emotion = face_emotion
            displayed_gender = gender_presentation

        cv2.putText(
            frame,
            "Age range: " + displayed_age_range,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Expression: " + displayed_emotion,
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Gender prediction: " + displayed_gender,
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "S: Speak | C: Clear | Q: Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SignBridge AI - Live Recognition",
            frame
        )

        # ------------------------------------------
        # KEYBOARD CONTROLS
        # ------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):

            if sentence:

                complete_sentence = " ".join(sentence)
                complete_sentence = complete_sentence.replace(
                    "_",
                    " "
                )

                print("Speaking sentence:", complete_sentence)

                engine.say(complete_sentence)
                engine.runAndWait()

            else:
                print("No sentence to speak.")

        elif key == ord("c"):

            sentence.clear()
            last_sign = None
            same_sign_count = 0
            prediction_history.clear()

            print("Sentence cleared.")

        elif key == ord("q"):
            break

finally:

    camera.release()
    hands.close()
    cv2.destroyAllWindows()

    print("SignBridge AI stopped.")

