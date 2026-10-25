
import cv2
from deepface import DeepFace

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot open camera.")
    exit()

print("SignBridge AI - Facial Analysis")
print("Press A to analyze your face.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Cannot read camera frame.")
        break

    # Display instructions on the camera feed
    cv2.putText(
        frame,
        "Press A: Analyze Face | Q: Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.imshow("SignBridge AI", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("a"):
        try:
            print("\nAnalyzing face... Please wait.")

            # Analyze age, facial expression, and gender presentation
            results = DeepFace.analyze(
                img_path=frame,
                actions=["age", "emotion", "gender"],
                detector_backend="opencv",
                enforce_detection=True,
                silent=True
            )

            face = results[0]

            # Extract results
            age = int(round(face["age"]))
            emotion = face["dominant_emotion"]

            gender_scores = face["gender"]
            predicted_gender = max(
                gender_scores,
                key=gender_scores.get
            )

            # Estimate an approximate age range
            if age <= 12:
                age_range = "0-12"
            elif age <= 19:
                age_range = "13-19"
            elif age <= 29:
                age_range = "20-29"
            elif age <= 39:
                age_range = "30-39"
            elif age <= 59:
                age_range = "40-59"
            else:
                age_range = "60+"

            # Print results in terminal
            print("\n========== RESULTS ==========")
            print("Estimated age:", age)
            print("Approximate age range:", age_range)
            print("Visible facial expression:", emotion)
            print("Predicted gender presentation:", predicted_gender)
            print("=============================")

        except Exception as error:
            print("Face analysis error:", error)

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

