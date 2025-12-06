Smart Attendance System using Python & OpenCV

A real-time, face-recognition-based automated attendance system built using Python 3.13, OpenCV, and LBPH Face Recognition.
The system detects faces, recognizes registered students, marks their attendance in a CSV file, and displays a visual confirmation along with a beep sound.

🚀 Features
🎯 Core Features

Real-time face detection using Haar Cascades

Face recognition using LBPH (Local Binary Patterns Histogram)

Automatic attendance marking into attendance.csv

Attendance logged only once per student

“PRESENT” visual popup on recognition

Beep sound feedback for successful marking

High accuracy model trained on multiple samples

Lightweight and works on low-end laptops

⚙️ Technical Highlights

Python 3.13 compatible

Uses OpenCV contrib face recognizer (LBPH)

Efficient dataset collection using webcam

Uses .gitignore to avoid pushing sensitive data

Clean and modular code (3 script structure)

🧩 Tech Stack
Technology	Purpose
Python 3.13	Main programming language
OpenCV & OpenCV-Contrib	Face detection & recognition
NumPy	Image processing
Pandas	Attendance storage
LBPH Algorithm	Face Recognition
Git/GitHub	Version control

📂 Project Structure
Smart-Attendance-System/
│── attendance.py         # Real-time attendance system
│── Capture_Faces.py      # Captures training images
│── train_model.py        # Trains LBPH face recognition model
│── .gitignore            # Ensures private data isn't uploaded
│
├── dataset/              # Face images (ignored in Git)
├── trainer.yml           # Trained model (ignored)
├── labels.pickle         # Name-ID map (ignored)
├── attendance.csv        # Attendance log (ignored)
└── venv/                 # Virtual environment (ignored)

📸 How It Works
1️⃣ Capture Faces
Run the script:
python Capture_Faces.py

Enter student name
System captures 10 high-quality images
Used for model training

2️⃣ Train the Model
python train_model.py
Trains LBPH recognizer
Saves trainer.yml + labels.pickle

3️⃣ Run the Attendance System
python attendance.py
The system will:

✔ Detect your face
✔ Recognize the student
✔ Display PRESENT in big text
✔ Play beep sound
✔ Mark attendance once

Example entry in attendance.csv:
name,time,date
SanskarBorate,17:33:12,2025-12-07

🧠 LBPH Algorithm in Short
LBPH (Local Binary Patterns Histogram) converts face features into numerical patterns:
Converts image to grayscale
Divides into small grids
Computes binary texture relationships
Creates histograms for identity
Matches using trained dataset
It is fast, lightweight, and works well with small datasets.

🛠️ How to Run Locally
Install dependencies
pip install opencv-python opencv-contrib-python numpy pandas

Run all scripts in this order
python Capture_Faces.py
python train_model.py
python attendance.py

🙌 Why This Project Is Useful

Works for schools, colleges, offices
Saves time vs manual attendance
Prevents proxy attendance
Can be extended into a full LMS system

🌟 Future Enhancements

GUI dashboard (Tkinter / PyQt)
Database integration (MySQL / Firebase)
Cloud sync attendance
Mobile app for teachers
Face registration module
Voice announcement (“Harsh Present!”)
Multi-camera classroom mode

👤 Author
Sanskar Tanaji Borate

GitHub: @sbcoding271

⭐ Support
If you found this project useful:

✔ Star the repo ⭐
✔ Share it
✔ Fork it
