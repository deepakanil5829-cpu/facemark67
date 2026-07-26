import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="📸",
    layout="wide"
)

# Load known face encodings and names
@st.cache_resource
def load_known_faces():

    photos_dir = os.path.join(os.path.dirname(__file__), "photos")

    images = {}

    for file in os.listdir(photos_dir):

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            name = os.path.splitext(file)[0].replace("_", " ").title()

            images[name] = os.path.join(photos_dir, file)

    encodings = []
    names = []

    for name, path in images.items():

        image = face_recognition.load_image_file(path)
        face_encoding = face_recognition.face_encodings(image)

        if len(face_encoding) == 0:
            st.warning(f"No face detected in {path}")
            continue

        encodings.append(face_encoding[0])
        names.append(name)

    return encodings, names


known_face_encoding, known_faces_names = load_known_faces()

if "attendance" not in st.session_state:
    st.session_state.attendance = []

class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
    
        self.present_students = []
        self.students = known_faces_names.copy()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        current_time = datetime.now()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        face_confidences = []

        for face_encoding in face_encodings:
            name = "Unknown"

            confidence = 0.0

            face_distance = face_recognition.face_distance(
                known_face_encoding,
                face_encoding
            )

            if len(face_distance) > 0:

                best_match_index = np.argmin(face_distance)

                if face_distance[best_match_index] < 0.50:

                    name = known_faces_names[best_match_index]

                    confidence = (1 - face_distance[best_match_index]) * 100

            face_names.append(name)
            face_confidences.append(confidence)
            if name in known_faces_names:
                
                if name in self.students:

                    attendance_time = current_time.strftime("%H:%M:%S")

                    self.present_students.append((name, attendance_time))

                    if not any(item["Name"] == name for item in st.session_state.attendance):
                        st.session_state.attendance.append({
                            "Name": name,
                            "Time": attendance_time
                        })

                    self.students.remove(name)

                    print(name, "Present")

        for (top, right, bottom, left), name, confidence in zip(
            face_locations,
            face_names,
            face_confidences
        ):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX

            if name == "Unknown":
                label = "Unknown"
            else:
                label = f"{name} ({confidence:.1f}%)"

            cv2.putText(
                img,
                label,
                (left + 6, bottom - 6),
                font,
                0.8,
                (255, 255, 255),
                1
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit app
st.title("📸 Face Recognition Attendance System")

if st.button("Reset Attendance"):
    st.session_state.attendance = []
    st.success("Attendance cleared.")

# Run the video processing function when 'Start Attendance' button is clicked
if "camera_started" not in st.session_state:
    st.session_state.camera_started = False

if st.button("Start Attendance"):
    st.session_state.camera_started = True

if st.session_state.camera_started:

    st.write("Starting attendance...")

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=FaceRecognitionProcessor,
        async_processing=True,
        media_stream_constraints={
            "video": {
                "width": 640,
                "height": 480,
                "frameRate": 15,
            },
            "audio": False,
        },
    )

    st.info("Allow camera permission and look at the camera.")

    st.write(
        "Today's Date:",
        datetime.now().strftime("%d-%m-%Y")
    )

else:
    st.write("Click the 'Start Attendance' button to begin.")

st.markdown("---")

if st.session_state.attendance:

    st.subheader("Attendance")

    df = pd.DataFrame(st.session_state.attendance)

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Attendance",
        csv,
        file_name=f"Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv",
        mime="text/csv"
    )
