import cv2
import numpy as np
import pickle
from datetime import datetime
import pandas as pd
import os
import winsound  # for beep sound

MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.pickle"
ATTENDANCE_FILE = "attendance.csv"

# Load trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

# Load labels
with open(LABELS_FILE, "rb") as f:
    label_ids = pickle.load(f)

# Reverse dictionary: id -> name
id_to_name = {v: k for k, v in label_ids.items()}

# Haar cascade for live detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Create attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["name", "time", "date"])
    df.to_csv(ATTENDANCE_FILE, index=False)

marked = set()  # Keep track of already marked students
last_mark_time = 0  # for displaying PRESENT text
show_present = False
present_name = ""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("[INFO] Running attendance system...")
print("[INFO] Press 'q' to exit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(120, 120))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        id_, confidence = recognizer.predict(roi_gray)

        # Recognize only if confidence < 80
        if confidence < 80:
            name = id_to_name.get(id_, "Unknown")
        else:
            name = "Unknown"

        # Draw rectangle around face
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(frame,
                    f"{name} ({int(confidence)})" if name != "Unknown" else "Unknown",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

        # Mark attendance & show "PRESENT"
        if name != "Unknown" and name not in marked:
            now = datetime.now()
            date = now.date().isoformat()
            time_ = now.strftime("%H:%M:%S")

            df = pd.DataFrame([[name, time_, date]], columns=["name", "time", "date"])
            df.to_csv(ATTENDANCE_FILE, mode="a", header=False, index=False)

            marked.add(name)
            present_name = name
            show_present = True
            last_mark_time = datetime.now().timestamp()

            # Play beep sound
            winsound.Beep(1000, 300)  # frequency 1000Hz, duration 300ms

            print(f"[MARKED] {name} at {time_} on {date}")

    # SHOW "PRESENT" TEXT FOR 2 SECONDS
    if show_present:
        if datetime.now().timestamp() - last_mark_time < 2:
            cv2.putText(frame,
                        f"{present_name} PRESENT!",
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        4)
        else:
            show_present = False

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n[INFO] Attendance session ended.")
print("[INFO] Students marked present:", list(marked))
