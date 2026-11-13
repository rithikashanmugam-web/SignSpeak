
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Gesture labels
GESTURES = {
    ord('1'): "A",
    ord('2'): "B",
    ord('3'): "C",
    ord('4'): "D",
    ord('5'): "E",
    ord('6'): "F"
}

# Dataset setup
os.makedirs("dataset", exist_ok=True)
CSV_FILE = "dataset/gestures.csv"

# CSV columns
columns = ["label"] + [
    f"feature_{i}" for i in range(63)
]

# Create CSV if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

# Normalize landmark coordinates
def extract_features(hand_landmarks):

    points = []

    for landmark in hand_landmarks.landmark:
        points.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    points = np.array(points, dtype=np.float32)

    # Make wrist the origin
    points = points - points[0]

    # Scale normalization
    scale = np.max(np.abs(points))

    if scale > 0:
        points = points / scale

    return points.flatten()

# Start hand detection
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam!")
    exit()

selected_gesture = None

print("\nSIGN SPEAK - DATA COLLECTION")
print("--------------------------------")
print("1 - Select A")
print("2 - Select B")
print("3 - Select C")
print("4 - Select D")
print("5 - Select E")
print("6 - Select F")
print("SPACE - Capture sample")
print("Q - Quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    hand_detected = False
    features = None

    if results.multi_hand_landmarks:

        hand_detected = True

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        features = extract_features(hand_landmarks)

    # Display instructions
    cv2.putText(
        frame,
        "Selected: " + str(selected_gesture),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    status = "Hand Detected" if hand_detected else "Show Your Hand"

    cv2.putText(
        frame,
        status,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "1-6: Select Gesture | SPACE: Capture | Q: Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.imshow("SignSpeak - Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Select gesture
    if key in GESTURES:
        selected_gesture = GESTURES[key]
        print("Selected gesture:", selected_gesture)

    # Capture sample
    elif key == 32:

        if selected_gesture is None:
            print("Select a gesture first!")

        elif not hand_detected:
            print("No hand detected!")

        else:
            with open(
                CSV_FILE, "a", newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    [selected_gesture] + features.tolist()
                )

            print(
                f"Captured sample for {selected_gesture}"
            )

    # Exit
    elif key == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

print("\nData collection completed!")
