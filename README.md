# Features

Data collection: Pose (33) + Face Mesh (468) landmarks → CSV (x, y, z, visibility for each).

Modeling: scikit-learn pipelines with StandardScaler; compares Logistic Regression, Ridge, RandomForest, GradientBoost; saves best (RandomForest) to body_language.pkl.

Realtime: Overlay predicted label + probability; trigger notifications (Notify) and audio (non-blocking, with cooldown).

Debounce/Cooldown: Prevents rapid repeat alerts per label.

Windows-friendly: Uses winsound and cv2.CAP_DSHOW for reliable camera access.

1) Requirements

Python 3.9–3.11 (Windows recommended)

Webcam

# Install dependencies

pip install opencv-python mediapipe numpy pandas scikit-learn notifypy

If you see MediaPipe build/runtime warnings, they’re safe to ignore.


2) Collect labeled data

Run collect() for each class you want (e.g., “Focused”, “Distracted”, “Not happy working”). Press q to stop a session.

from face import collect

# Collect samples for each class label you care about:
collect(class_name="Focused",     output_csv="data/coords.csv")
collect(class_name="Distracted",  output_csv="data/coords.csv")
collect(class_name="Not happy working", output_csv="data/coords.csv")

You can either delete everything except for face.py and collect your own data, or you can use my own.

Tips:

Sit centered, good lighting, keep your face visible.

Capture a few minutes per class (varied angles/expressions) to reduce overfitting.

3) Train the model
from face import train
train(csv_path="data/coords.csv", model_path="models/body_language.pkl")


You’ll see accuracy printed per algorithm; the script saves the RandomForest pipeline by default.

4) Run real-time predictions
from face import realtime
realtime(model_path="models/body_language.pkl")


A window titled Prediction will open with:

Top-left overlay: label + confidence.

Actions fire based on the predicted label (see below).

Press q to quit.

Customizing actions & audio

Actions are defined in realtime() inside the trigger_action(label) function.

Add or modify elif blocks for your labels.

Place .wav files in sounds/ and update paths accordingly.

Cooldown (cooldown_time) controls how often a label may trigger again (default 5 seconds).




# Acknowledgments

MediaPipe Holistic & Face Mesh (Google)

OpenCV

scikit-learn

NotifyPy
