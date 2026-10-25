# 🤟 Automatic Sign Language Conversion

An AI-powered application that recognizes predefined hand gestures in real time and converts them into text and speech. The project combines computer vision, machine learning, and facial analysis to create an interactive sign language recognition system.

## 📌 About the Project

The Automatic Sign Language Conversion project aims to make communication easier by recognizing trained hand gestures through a webcam. Detected gestures are converted into text, and the resulting text can be spoken aloud using text-to-speech functionality.

The project also includes an optional facial analysis feature that estimates facial attributes and expressions.

**Note:** The system recognizes the gestures represented in its training dataset. It is not a universal translator for every sign language.

## ✨ Features

* **Real-Time Gesture Recognition:** Detects and classifies trained hand gestures using a webcam.
* **Two-Hand Detection:** Supports recognition using hand landmark information from both hands.
* **Text Conversion:** Converts recognized gestures into readable text.
* **Text-to-Speech:** Uses speech synthesis to speak recognized text.
* **Facial Analysis:** Provides estimated age, gender category, and facial expression analysis.
* **Graphical User Interface:** Provides an interactive interface for using the system.
* **Custom Dataset:** Supports collecting gesture samples to train the machine learning model.
* **Machine Learning:** Uses a trained classification model to predict gestures.

## 🛠️ Technologies Used

| Technology    | Purpose                                        |
| ------------- | ---------------------------------------------- |
| Python        | Core programming language                      |
| OpenCV        | Webcam access and image processing             |
| MediaPipe     | Hand landmark detection                        |
| NumPy         | Numerical operations and feature processing    |
| Pandas        | Dataset handling                               |
| Scikit-learn  | Machine learning model training and evaluation |
| Joblib        | Saving and loading the trained model           |
| pyttsx3       | Text-to-speech functionality                   |
| DeepFace      | Facial attribute and expression analysis       |
| CustomTkinter | Graphical user interface                       |

## 🤟 Supported Gestures

The model is designed to recognize the following trained gesture classes:

HELLO, NAMASTE, GOOD_MORNING, GOOD_NIGHT, GOODBYE, YES, NO, PLEASE, SORRY, THANK_YOU, WELCOME, HELP, STOP, COME, GO, EAT, DRINK, WATER, FOOD, SLEEP, HOME, SCHOOL, FRIEND, FAMILY, HAPPY, SAD, LOVE, UNDERSTAND, GOOD, BAD.

Recognition performance depends on the quality of the training data, hand positioning, lighting, and the similarity between gestures.

## 📂 Project Structure

```text
Automatic-Sign-Language-Conversion/
│
├── collect_data.py
├── face_analysis.py
├── real_time_recognition.py
├── signbridge_gui.py
├── sign_language_conversion.py
├── train_model.py
├── sign_model.pkl
├── sign_data.csv
├── sign_data_2hand.csv
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/239viraj/Automatic-Sign-Language-Conversion.git
```

### 2. Open the project folder

```bash
cd Automatic-Sign-Language-Conversion
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

Install the required Python packages:

```bash
pip install opencv-contrib-python-headless==4.10.0.84 numpy==1.26.4 mediapipe==0.10.21 pandas scikit-learn joblib pyttsx3 customtkinter deepface
```

Package versions and compatibility may need adjustment for a different Python environment.

### 5. Run the application

To start the graphical application:

```bash
python signbridge_gui.py
```

Make sure your webcam is connected and accessible.

## 🚀 Other Available Scripts

### Collect gesture training data

```bash
python collect_data.py
```

### Train the gesture recognition model

```bash
python train_model.py
```

### Run real-time gesture recognition

```bash
python real_time_recognition.py
```

The facial analysis functionality is also available in `face_analysis.py`.

## 🧠 How It Works

1. The webcam captures a live video stream.
2. MediaPipe detects hand landmarks in the video.
3. Landmark coordinates are converted into numerical features.
4. The trained machine learning model predicts the corresponding gesture.
5. The recognized gesture is displayed as text.
6. Text-to-speech functionality can speak the recognized output.
7. The optional facial analysis component estimates facial attributes and expressions.

## 🔮 Future Improvements

* Expand the gesture vocabulary.
* Improve recognition under different lighting conditions.
* Add support for more complex gestures and continuous sentences.
* Improve model generalization using more varied training samples.
* Enhance the interface and provide more feedback during recognition.
* Explore additional sign languages and gesture datasets.

## ⚠️ Limitations

* The model recognizes only the gesture classes represented in its training data.
* Prediction accuracy depends on training-data quality and recording conditions.
* Some gestures may require specific hand positions or both hands.
* Facial analysis results are estimates and may be inaccurate.
* The application is a prototype and should not replace a qualified interpreter in situations where accurate communication is essential.

## 👨‍💻 Author

**Viraj Kumar**

## 📄 License

No license has been specified yet. Contact the repository owner for permission before redistributing or reusing this project's code.
