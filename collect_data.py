
import cv2
import mediapipe as mp
import csv
import os
import time

# ==========================================
# SETTINGS
# ==========================================

SIGNS = [
    "HELLO",
    "NAMASTE",
    "GOOD_MORNING",
    "GOOD_NIGHT",
    "GOODBYE",
    "YES",
    "NO",
    "PLEASE",
    "SORRY",
    "THANK_YOU"
    "WELCOME",
    "HELP",
    "STOP",
    "COME",
    "GO",
    "EAT",
    "DRINK",
    "WATER",
    "FOOD",
    "SLEEP",
    "HOME",
    "SCHOOL",
    "FRIEND",
    "FAMILY",
    "HAPPY",
    "SAD",
    "LOVE",
    "UNDERSTAND",
    "GOOD",
    "BAD"
]

SAMPLES_NEEDED = 300

# Keep the original 63-feature dataset untouched.
FILE_NAME = "sign_data_2hand.csv"

# ==========================================
# CSV HEADER: 126 FEATURES + LABEL
# ==========================================

header = []

for hand in ["left", "right"]:
    for i in range(21):
        header.extend([
            f"{hand}_x{i}",
            f"{hand}_y{i}",
            f"{hand}_z{i}"
        ])

header.append("label")

# Create a new dataset or validate an existing one.
if os.path.exists(FILE_NAME):
    with open(FILE_NAME, "r", newline="") as check_file:
        reader = csv.reader(check_file)
        existing_header = next(reader, None)

    if existing_header != header:
        print("ERROR: Existing two-hand CSV has an incorrect header.")
        print("No data was collected.")
        raise SystemExit
else:
    with open(FILE_NAME, "w", newline="") as new_file:
        csv.writer(new_file).writerow(header)

# ==========================================
# MEDIAPIPE SETUP
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==========================================
# CAMERA SETUP
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    hands.close()
    raise SystemExit

# ==========================================
# HELPER: EXTRACT HAND FEATURES
# ==========================================

def get_hand_features(results):
    left_features = [0.0] * 63
    right_features = [0.0] * 63

    if not results.multi_hand_landmarks:
        return None

    handedness_results = results.multi_handedness or []

    for index, landmarks in enumerate(results.multi_hand_landmarks):

        mp_draw.draw_landmarks(
            frame,
            landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        features = []

        for point in landmarks.landmark:
            features.extend([point.x, point.y, point.z])

        if index >= len(handedness_results):
            continue

        classifications = handedness_results[index].classification

        if not classifications:
            continue

        label = classifications[0].label

        # The camera frame is mirrored before MediaPipe processing.
        # Swap labels to maintain the user's anatomical left/right order.
        if label == "Left":
            label = "Right"
        elif label == "Right":
            label = "Left"

        if label == "Left":
            left_features = features
        elif label == "Right":
            right_features = features

    # Both hands absent means there is no valid sample.
    if (
        left_features == [0.0] * 63
        and right_features == [0.0] * 63
    ):
        return None

    return left_features + right_features

# ==========================================
# DATA COLLECTION
# ==========================================

total_signs = len(SIGNS)

try:
    with open(FILE_NAME, "a", newline="") as file:
        writer = csv.writer(file)

        for sign_index, sign in enumerate(SIGNS, start=1):
            sample_count = 0

            print("\n================================")
            print(f"Sign {sign_index}/{total_signs}: {sign}")
            print(f"Target samples: {SAMPLES_NEEDED}")
            print("================================")

            # Preparation countdown.
            start_time = time.time()

            while time.time() - start_time < 3:
                success, frame = camera.read()

                if not success:
                    raise RuntimeError("Could not read from camera.")

                frame = cv2.flip(frame, 1)

                remaining = max(
                    0,
                    3 - int(time.time() - start_time)
                )

                cv2.putText(
                    frame,
                    f"Prepare: {sign}",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Starting in: {remaining}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Q = Quit",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("SignBridge AI - Data Collection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise SystemExit

            # Collect samples for this sign.
            while sample_count < SAMPLES_NEEDED:
                success, frame = camera.read()

                if not success:
                    raise RuntimeError("Could not read from camera.")

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = hands.process(rgb_frame)

                features = get_hand_features(results)

                if features is not None:
                    writer.writerow(features + [sign])
                    sample_count += 1

                    if sample_count % 25 == 0:
                        file.flush()

                cv2.putText(
                    frame,
                    f"Sign: {sign}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Samples: {sample_count}/{SAMPLES_NEEDED}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Sign: {sign_index}/{total_signs}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Keep required hand(s) visible | Q = Quit",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("SignBridge AI - Data Collection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    file.flush()
                    raise SystemExit

            file.flush()

            print(
                f"{sign}: {sample_count}/{SAMPLES_NEEDED} samples collected."
            )

    print("\n================================")
    print("DATA COLLECTION COMPLETED!")
    print(f"New signs: {total_signs}")
    print(f"Target samples: {total_signs * SAMPLES_NEEDED}")
    print(f"Saved to: {FILE_NAME}")
    print("Original sign_data.csv was not modified.")
    print("================================")

except SystemExit:
    print("\nCollection stopped.")

except Exception as error:
    print(f"\nCollection stopped because of an error: {error}")

finally:
    camera.release()
    hands.close()
    cv2.destroyAllWindows()

