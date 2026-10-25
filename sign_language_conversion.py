import cv2
import mediapipe as mp

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open camera
camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        print("Cannot read camera")
        break

    # OpenCV uses BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    # Check if hand is detected
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Print 21 landmark coordinates
            for i, landmark in enumerate(hand_landmarks.landmark):

                print(
                    f"Landmark {i}: "
                    f"x={landmark.x:.4f}, "
                    f"y={landmark.y:.4f}, "
                    f"z={landmark.z:.4f}"
                )

            print("------------------------")

    # Display camera
    cv2.imshow("Sign Language Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows() 