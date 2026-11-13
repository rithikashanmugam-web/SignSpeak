
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json

from collections import deque

# -----------------------------
# 1. MODEL CONFIGURATION
# -----------------------------

DEVICE = torch.device("cpu")

INPUT_SIZE = 1662
HIDDEN_SIZE = 64
NUM_CLASSES = 40
SEQUENCE_LENGTH = 30

# -----------------------------
# 2. DEFINE LSTM MODEL
# -----------------------------

class ISL_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        output = hidden[-1]
        output = self.dropout(output)

        return self.fc(output)


# -----------------------------
# 3. LOAD TRAINED MODEL
# -----------------------------

checkpoint = torch.load(
    "models/isl_phrase_lstm.pth",
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]

model = ISL_LSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_classes=len(class_names)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("ISL Phrase Model Loaded Successfully!")


# -----------------------------
# 4. MEDIAPIPE HOLISTIC
# -----------------------------

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# 5. EXTRACT 1662 FEATURES
# -----------------------------

def extract_keypoints(results):

    pose = np.array([
        [lm.x, lm.y, lm.z, lm.visibility]
        for lm in results.pose_landmarks.landmark
    ]).flatten() if results.pose_landmarks else np.zeros(132)

    face = np.array([
        [lm.x, lm.y, lm.z]
        for lm in results.face_landmarks.landmark
    ]).flatten() if results.face_landmarks else np.zeros(1404)

    left_hand = np.array([
        [lm.x, lm.y, lm.z]
        for lm in results.left_hand_landmarks.landmark
    ]).flatten() if results.left_hand_landmarks else np.zeros(63)

    right_hand = np.array([
        [lm.x, lm.y, lm.z]
        for lm in results.right_hand_landmarks.landmark
    ]).flatten() if results.right_hand_landmarks else np.zeros(63)

    return np.concatenate([
        pose,
        face,
        left_hand,
        right_hand
    ])


# -----------------------------
# 6. START WEBCAM
# -----------------------------

sequence = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam!")

prediction_text = "Waiting for signs..."
confidence_text = ""

print("Webcam started. Press Q to quit.")


# -----------------------------
# 7. LIVE RECOGNITION LOOP
# -----------------------------

with torch.no_grad():

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Unable to read webcam frame.")
            break

        # Mirror the webcam preview
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect landmarks
        results = holistic.process(image)

        image.flags.writeable = True

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS
        )

        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

        # Extract features
        keypoints = extract_keypoints(results)

        sequence.append(keypoints)

        # Predict after collecting 30 frames
        if len(sequence) == SEQUENCE_LENGTH:

            input_data = np.expand_dims(
                np.array(sequence, dtype=np.float32),
                axis=0
            )

            input_tensor = torch.tensor(
                input_data,
                dtype=torch.float32
            ).to(DEVICE)

            outputs = model(input_tensor)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )

            confidence_value = confidence.item()

            if confidence_value >= 0.50:

                prediction_text = class_names[
                    predicted_index.item()
                ]

                confidence_text = (
                    f"Confidence: {confidence_value * 100:.2f}%"
                )

            else:
                prediction_text = "Uncertain"
                confidence_text = (
                    f"Confidence: {confidence_value * 100:.2f}%"
                )

        # Display prediction
        cv2.rectangle(
            frame,
            (0, 0),
            (640, 100),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Prediction: {prediction_text}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            confidence_text,
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.imshow("SignSpeak - ISL Phrase Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# -----------------------------
# 8. RELEASE RESOURCES
# -----------------------------

cap.release()
holistic.close()
cv2.destroyAllWindows()

print("SignSpeak webcam closed.")
