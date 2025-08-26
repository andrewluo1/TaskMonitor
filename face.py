import cv2
import csv
import os
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
import winsound
import threading
import wave
from notifypy import Notify

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

def collect(class_name="Default", output_csv="coords.csv"):
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        if not os.path.exists(output_csv):
            num_coords = 33 + 468
            landmarks = ['class']
            for val in range(1, num_coords + 1):
                landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
            with open(output_csv, mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.face_landmarks:
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            try:
                if results.pose_landmarks and results.face_landmarks:
                    pose = results.pose_landmarks.landmark
                    face = results.face_landmarks.landmark
                    pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose]).flatten())
                    face_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in face]).flatten())
                    row = [class_name] + pose_row + face_row
                    with open(output_csv, mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row)
            except Exception as e:
                print(f"Row write error: {e}")

            cv2.imshow('Collecting Landmarks', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release
    cv2.destroyAllWindows()


def train(csv_path="coords.csv", model_path="body_language.pkl"):
    df = pd.read_csv(csv_path)
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    pipelines = {
        'logisticreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        'ridgeClassifier': make_pipeline(StandardScaler(), RidgeClassifier()),
        'randomForest': make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gradientBoost': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }
    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model
        yhat = model.predict(X_test)
        print(algo, accuracy_score(y_test, yhat))
    with open(model_path, 'wb') as f:
        pickle.dump(fit_models['randomForest'], f)

class AsyncWavPlayer:
    def __init__(self):
        self._lock = threading.Lock()
        self._is_playing = False
        self._stop_timer = None
    def _duration_sec(self, path):
        try:
            with wave.open(path, 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                return frames / float(rate) if rate else 0.0
        except Exception:
            return 2.0
    def play(self, path):
        winsound.PlaySound(None, winsound.SND_PURGE)
        dur = self._duration_sec(path)
        with self._lock:
            if self._is_playing:
                return 
            self._is_playing = True
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            if self._stop_timer is not None:
                self._stop_timer.cancel()
            self._stop_timer = threading.Timer(dur + 0.05, self._mark_done)
            self._stop_timer.start()

    def _mark_done(self):
        with self._lock:
            self._is_playing = False

player = AsyncWavPlayer()

def realtime(model_path="body_language.pkl"):
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    cooldown = {}
    cooldown_time = 5


    def trigger_action(label):
        if label == "Distracted":
            #sound_path = os.path.abspath("backtowork.wav")
            #player.play(sound_path)
            notification = Notify()
            notification.title = "DETECTED BEING DISTRACTED"
            notification.message = "BACK TO WORK BUDDY"
            notification.audio = "backtowork.wav"
            notification.send()

        elif label == "Not happy working":
            sound_path2 = os.path.abspath("incorrect.wav")
            player.play(sound_path2)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    with mp_holistic.Holistic(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks and results.face_landmarks:
                    pose = results.pose_landmarks.landmark
                    face = results.face_landmarks.landmark
                    pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose]).flatten())
                    face_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in face]).flatten())
                    row = pose_row + face_row
                    X = pd.DataFrame([row])

                    label = model.predict(X)[0]
                    prob = model.predict_proba(X)[0]

                    cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
                    cv2.putText(image, label, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"{round(np.max(prob), 2)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    now = time.time()
                    last_trigger = cooldown.get(label, 0)
                    if now - last_trigger >= cooldown_time:
                        trigger_action(label)
                        cooldown[label] = now

            except Exception as e:
                print(f"[WARN] Prediction failed: {e}")

            cv2.imshow('Prediction', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

