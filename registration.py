import cv2
import sqlite3
import numpy as np
import os
import face_recognition

DB_PATH = "attendance.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            semester TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
    ''')
    conn.commit()
    conn.close()

def register_student():
    init_db()
    print("\n--- FACEMARK STUDENT REGISTRATION ---")
    student_id = input("Enter Student ID (e.g., MCT24AIM014): ").strip()
    name = input("Enter Full Name: ").strip()
    dept = input("Enter Department: ").strip()
    sem = input("Enter Semester: ").strip()

    if not student_id or not name:
        print("❌ Student ID and Name are required!")
        return

    # Check if student already exists
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT student_id FROM students WHERE student_id = ?", (student_id,))
    if cursor.fetchone():
        print(f"⚠️ Student ID {student_id} already exists in the database.")
        conn.close()
        return

    # Start Video Capture
    cap = cv2.VideoCapture(0)
    print("\n📷 Opening Webcam... Stand in front of the camera.")
    print("Press 'c' to capture a face sample (Capture 3-5 samples for better accuracy).")
    print("Press 'q' when done.")

    embeddings = []
    samples_captured = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame from camera.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        # Draw bounding boxes around detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        status_text = f"Samples Captured: {samples_captured} | Press 'c' to capture, 'q' to finish"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("FaceMark Registration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(face_locations) == 0:
                print("⚠️ No face detected in frame. Adjust position and try again.")
            elif len(face_locations) > 1:
                print("⚠️ Multiple faces detected! Ensure only ONE person is in frame.")
            else:
                encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if encodings:
                    embeddings.append(encodings[0])
                    samples_captured += 1
                    print(f"✓ Captured sample #{samples_captured}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == 0:
        print("❌ No face samples were captured. Registration cancelled.")
        conn.close()
        return

    # Save Student Metadata
    cursor.execute("INSERT INTO students (student_id, name, department, semester) VALUES (?, ?, ?, ?)",
                   (student_id, name, dept, sem))

    # Save Face Embeddings (128-d floats stored as raw byte blobs)
    for emb in embeddings:
        emb_bytes = emb.astype(np.float64).tobytes()
        cursor.execute("INSERT INTO face_embeddings (student_id, embedding) VALUES (?, ?)",
                       (student_id, emb_bytes))

    conn.commit()
    conn.close()
    print(f"\n✅ Successfully registered {name} ({student_id}) with {len(embeddings)} face samples!")

if __name__ == "__main__":
    register_student()