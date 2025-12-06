import cv2
import os
import time

# Ask for student name
student_name = input("Enter student name (no spaces, e.g., SanskarBorate): ").strip()

if not student_name:
    raise ValueError("Name cannot be empty")

DATASET_DIR = "dataset"
person_dir = os.path.join(DATASET_DIR, student_name)
os.makedirs(person_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

max_images = 10         # Capture only 10 photos
count = 0               # number of captured images
last_capture_time = 0   # for 1-second delay between captures

print("\n[INFO] Look at the camera. A photo will be taken every 1 second.")
print("[INFO] Press 'q' anytime to cancel.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(120, 120)
    )

    # Draw rectangle for visual feedback
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Capture only when:
    # 1. Face detected
    # 2. At least 1 second passed since last photo
    current_time = time.time()
    if len(faces) > 0 and (current_time - last_capture_time) >= 1:
        (x, y, w, h) = faces[0]  # take first detected face
        face_roi = gray[y:y + h, x:x + w]

        count += 1
        img_path = os.path.join(person_dir, f"img_{count}.jpg")
        cv2.imwrite(img_path, face_roi)
        last_capture_time = current_time

        print(f"[CAPTURED {count}/10] {img_path}")

        # Show capture overlay
        cv2.putText(frame, f"Captured {count}/10", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Stop after collecting 10 images
    if count >= max_images:
        print("\n[INFO] Successfully captured 10 images!")
        break

    cv2.imshow("Capture Faces (Auto)", frame)

    # Allow manual exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Capture cancelled by user.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Saved images to folder: {person_dir}")
