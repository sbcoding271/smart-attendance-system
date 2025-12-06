import cv2
import os
import numpy as np
import pickle

DATASET_DIR = "dataset"
MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.pickle"

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# LBPH recognizer (OpenCV contrib)
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}      # name -> numeric ID
faces = []          # face ROIs
ids = []            # labels (IDs)

print("\n[INFO] Starting training...")

# walk through dataset folder
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith(("png", "jpg", "jpeg")):

            path = os.path.join(root, file)
            name = os.path.basename(root)  # folder name = student name

            # assign each folder a numeric ID
            if name not in label_ids:
                label_ids[name] = current_id
                current_id += 1

            id_ = label_ids[name]

            # read image in grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Could not read {path}, skipping.")
                continue

            # detect face region inside image
            faces_rect = face_cascade.detectMultiScale(
                img,
                scaleFactor=1.3,
                minNeighbors=3
            )

            # if face not detected, still use entire image
            if len(faces_rect) == 0:
                faces.append(img)
                ids.append(id_)
            else:
                for (x, y, w, h) in faces_rect:
                    roi = img[y:y + h, x:x + w]
                    faces.append(roi)
                    ids.append(id_)

print(f"[INFO] Total faces found: {len(faces)}")
print(f"[INFO] Students detected: {len(label_ids)}")

if len(faces) == 0:
    raise RuntimeError("No faces found! Run capture_faces.py first.")

# train LBPH model
recognizer.train(faces, np.array(ids))
recognizer.save(MODEL_FILE)

# save label mapping
with open(LABELS_FILE, "wb") as f:
    pickle.dump(label_ids, f)

print("[INFO] Training complete!")
print(f"[INFO] Model saved as: {MODEL_FILE}")
print(f"[INFO] Labels saved as: {LABELS_FILE}\n")
