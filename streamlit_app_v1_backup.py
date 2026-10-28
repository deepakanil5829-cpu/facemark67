import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
import database
from datetime import datetime


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FaceMark",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ================= FACE MARK UI THEME =================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080D19;
    --sidebar: #0D1525;
    --panel: #111B2E;
    --panel-hover: #172640;
    --blue: #398BFF;
    --blue-light: #72B5FF;
    --text: #F1F5FC;
    --muted: #91A1B9;
    --border: #263650;
    --green: #32D6A0;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(8, 13, 25, 0.95);
}

[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #C5D1E3;
}

h1, h2, h3 {
    color: var(--text) !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

p, label {
    color: #B2C0D4;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #15233B, #101929);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-weight: 800;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    background: #0D1729;
    color: white;
    border: 1px solid #30435F;
    border-radius: 9px;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #0D1729;
    color: white;
    border-color: #30435F;
    border-radius: 9px;
}

.stButton > button,
[data-testid="stFormSubmitButton"] button {
    background: var(--blue);
    color: white;
    border: 1px solid var(--blue);
    border-radius: 9px;
    min-height: 42px;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #2375E8;
    border-color: var(--blue-light);
    color: white;
    box-shadow: 0 0 16px #398BFF35;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 12px;
}

[data-testid="stExpander"],
[data-testid="stForm"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

hr {
    border-color: var(--border);
}

#MainMenu,
footer {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    [data-testid="stMetric"] {
        padding: 13px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CUSTOM CSS
# =========================================================

# ================= UI HELPER FUNCTIONS =================

def fm_page_header(title, subtitle):
    st.markdown(
        f"""
        <div style="
            padding: 8px 0 22px 0;
            border-bottom: 1px solid #263650;
            margin-bottom: 24px;
        ">
            <div style="
                font-size: 30px;
                font-weight: 800;
                color: #F1F5FC;
                letter-spacing: -0.8px;
            ">{title}</div>
            <div style="
                font-size: 14px;
                color: #91A1B9;
                margin-top: 8px;
            ">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def fm_section(title, subtitle=""):
    st.markdown(
        f"""
        <div style="margin: 14px 0 18px 0;">
            <div style="
                color: #F1F5FC;
                font-size: 19px;
                font-weight: 700;
            ">{title}</div>
            <div style="
                color: #91A1B9;
                font-size: 13px;
                margin-top: 5px;
            ">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# LOAD KNOWN FACES
# =========================================================

@st.cache_resource

def load_known_faces():
    encodings = []
    names = []

    try:
        # Get currently registered students
        students = database.get_students()
        face_data = database.get_all_face_embeddings()

        # Map student IDs to their names
        student_names = {
            str(student["student_id"]): student["name"]
            for student in students
        }

        # Load embeddings only for registered students
        for item in face_data:
            student_id = str(item["student_id"])
            embedding = item["embedding"]

            if student_id in student_names:
                encodings.append(np.asarray(embedding))
                names.append(student_names[student_id])

    except Exception as e:
        st.error(f"Could not load database face embeddings: {e}")

    return encodings, names


known_face_encoding, known_faces_names = load_known_faces()

# =========================================================
# SESSION STATE
# =========================================================

if "attendance" not in st.session_state:
    st.session_state.attendance = []

if "camera_started" not in st.session_state:
    st.session_state.camera_started = False


# =========================================================
# FACE RECOGNITION PROCESSOR
# =========================================================

class FaceRecognitionProcessor(VideoProcessorBase):

    def __init__(self):
     self.present_students = []
     self.students = known_faces_names.copy()
     self.frame_count = 0
     self.last_face_names = []
     self.last_face_confidences = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        current_time = datetime.now()
        self.frame_count += 1

        if self.frame_count % 5 != 0:
          return frame

        img = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror camera
        #img = cv2.flip(img, 1)

        # Reduce image size for faster processing
        small_frame = cv2.resize(
            img,
            (0, 0),
            fx=0.20,
            fy=0.20
        )

        rgb_small_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )
        print("Starting face detection")

        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_small_frame
        )
        print("Face detection finished:", len(face_locations))

        # Generate face encodings
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame,
            face_locations
        )
        print("Face encoding finished:", len(face_encodings))

        face_names = []
        face_confidences = []

        for face_encoding in face_encodings:

            name = "Unknown"
            confidence = 0.0

            face_distance = face_recognition.face_distance(
                known_face_encoding,
                face_encoding
            )
            print("Loaded known faces:", len(known_face_encoding))
            print("Face distances:", face_distance)
            if len(face_distance) > 0:

                best_match_index = np.argmin(
                    face_distance
                )

                if face_distance[best_match_index] < 0.50:

                    name = known_faces_names[
                        best_match_index
                    ]

                    confidence = (
                        1 -
                        face_distance[best_match_index]
                    ) * 100
                    print("Matched name:", name)
                    print("Confidence:", confidence)

            face_names.append(name)
            face_confidences.append(confidence)

            # Attendance
 
           
            if name in known_faces_names:
                if name in self.students:
                    attendance_time = current_time.strftime("%H:%M:%S")

                    if not any(
                        item[0] == name
                        for item in self.present_students
                    ):
                        self.present_students.append(
                            (name, attendance_time)
                        )
                        print(
                            "Attendance recorded in processor:",
                            self.present_students
                        )

                    self.students.remove(name)


        # Draw results on video
        for (
            top,
            right,
            bottom,
            left
        ), name, confidence in zip(
            face_locations,
            face_names,
            face_confidences
        ):

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(
                img,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                img,
                (left, bottom - 35),
                (right, bottom),
                (0, 255, 0),
                cv2.FILLED
            )

            if name == "Unknown":

                label = "Unknown"

            else:

                label = (
                    f"{name} "
                    f"({confidence:.1f}%)"
                )

            cv2.putText(
                img,
                label,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                1
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "# 🟣 FaceMark"
    )

    st.caption(
        "AI-Based Face Recognition Attendance System"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "📷 Attendance",
            "👨‍🎓 Students",
            "📋 Attendance Records"
        ]
    )

    st.markdown("---")

    st.caption(
        "Computer Vision Project"
    )

    st.caption(
        "FaceMark v1.0"
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="facemark-title">🟣 FaceMark</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="facemark-subtitle">'
    'AI-Based Face Recognition Attendance System'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.markdown(
        '<div class="section-title">Dashboard</div>',
        unsafe_allow_html=True
    )

    total_students = len(database.get_students())

    present_students = len(
        st.session_state.attendance
    )

    current_date = datetime.now().strftime(
        "%d-%m-%Y"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-number">
                    {total_students}
                </div>
                <div class="card-title">
                    Registered Students
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-number">
                    {present_students}
                </div>
                <div class="card-title">
                    Present Today
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-number">
                    {current_date}
                </div>
                <div class="card-title">
                    Today's Date
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("How FaceMark Works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info(
            "📷\n\n"
            "**1. Capture**\n\n"
            "Camera captures the face."
        )

    with col2:
        st.info(
            "🧠\n\n"
            "**2. Analyze**\n\n"
            "Face encoding is generated."
        )

    with col3:
        st.info(
            "🔍\n\n"
            "**3. Recognize**\n\n"
            "Encoding is compared with registered faces."
        )

    with col4:
        st.info(
            "✅\n\n"
            "**4. Mark**\n\n"
            "Attendance is recorded automatically."
        )

# =========================================================
# ATTENDANCE
# =========================================================

elif page == "📷 Attendance":

    st.markdown(
        '<div class="section-title">'
        '📷 Mark Attendance'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Allow camera access and look directly at the camera."
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button(
            "🚀 Start Attendance",
            use_container_width=True
        ):
            st.session_state.camera_started = True

    with col2:
        if st.button(
            "🔄 Reset",
            use_container_width=True
        ):
            st.session_state.attendance = []
            st.session_state.camera_started = False
            st.rerun()

    st.markdown("---")

    if st.session_state.camera_started:

        webrtc_ctx = webrtc_streamer(
            key="facemark-camera",
            video_processor_factory=FaceRecognitionProcessor,
            async_processing=True,
            media_stream_constraints={
                "video": {
                    "width": 640,
                    "height": 480,
                    "frameRate": 15
                },
                "audio": False
            }
        )
        st.session_state["facemark_webrtc_ctx"] = webrtc_ctx

        if webrtc_ctx.video_processor is not None:
            processor = webrtc_ctx.video_processor

            print("Processor available: True")
            print(
                "Processor attendance:",
                processor.present_students
            )

            if webrtc_ctx.state.playing:
                for name, attendance_time in list(
                    processor.present_students
                ):
                    if not any(
                        item["Name"] == name
                        for item in st.session_state.attendance
                    ):
                        st.session_state.attendance.append({
                            "Name": name,
                            "Time": attendance_time,
                            "Status": "Present"
                        })
        else:
            print("Processor is not available yet.")

        st.caption(
            "Click START inside the camera window "
            "to activate the webcam."
        )

    else:
        st.info(
            "Click **Start Attendance** to activate the camera."
        )

    st.markdown("---")

    
    @st.fragment(run_every="2s")
    def attendance_table():
        if st.session_state.camera_started:
            webrtc_ctx = st.session_state.get(
                "facemark_webrtc_ctx"
            )

            if webrtc_ctx is not None:
                processor = webrtc_ctx.video_processor

                if (
                    processor is not None
                    and webrtc_ctx.state.playing
                ):
                    for name, attendance_time in list(
                        processor.present_students
                    ):
                        if not any(
                            item["Name"] == name
                            for item in st.session_state.attendance
                        ):
                            st.session_state.attendance.append({
                                "Name": name,
                                "Time": attendance_time,
                                "Status": "Present"
                            })

        st.subheader("Today's Attendance")

        if st.session_state.attendance:
            df = pd.DataFrame(st.session_state.attendance)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No attendance has been marked yet.")

    attendance_table()



# =========================================================
# STUDENTS
# =========================================================

elif page == "👨‍🎓 Students":

    st.markdown(
        '<div class="section-title">'
        '👨‍🎓 Registered Students'
        '</div>',
        unsafe_allow_html=True
    )


    st.subheader("➕ Register a Student")

    with st.form("register_student_form"):
        student_id = st.text_input("Student ID *")
        name = st.text_input("Full Name *")
        department = st.text_input("Department")
        semester = st.text_input("Semester")

        st.caption(
            "Enter the details, then submit the form. "
            "You can capture the student's face in the next step."
        )

        submitted = st.form_submit_button(
            "Continue to Face Capture",
            use_container_width=True
        )

    if submitted:
        if not student_id.strip() or not name.strip():
            st.error("Student ID and Full Name are required.")

        elif database.get_student(student_id.strip()):
            st.error("This Student ID already exists.")

        else:
            st.session_state["pending_student"] = {
                "student_id": student_id.strip(),
                "name": name.strip(),
                "department": department.strip(),
                "semester": semester.strip()
            }
            st.rerun()

    pending = st.session_state.get("pending_student")


    if pending:

     st.divider()
     st.subheader("📸 Capture Student Face")
     st.write(f"Registering face for **{pending['name']}**")

    camera_photo = None

    capture_method = st.radio(
        "Choose face registration method",
        ["Upload phone photo", "Use webcam"],
        horizontal=True
    )

    if capture_method == "Upload phone photo":
        camera_photo = st.file_uploader(
            "Upload a photo from your phone",
            type=["jpg", "jpeg", "png"],
            key="student_face_upload"
        )
    else:
        camera_photo = st.camera_input(
            "Take a clear photo of the student's face",
            key="student_face_camera"
        )

    if camera_photo is not None:
        # Keep your existing Save Student and Face code here,
        # properly indented inside this if block.


     if camera_photo is not None:

            if st.button("Save Student and Face", type="primary"):

                import os
                from pathlib import Path
                from PIL import Image

                image = Image.open(camera_photo).convert("RGB")
                image_array = np.array(image)

                face_locations = face_recognition.face_locations(
                    image_array,
                    model="hog"
                )

                if len(face_locations) != 1:
                    st.error(
                        "Please capture a photo containing exactly "
                        "one clearly visible face."
                    )

                else:
                    encodings = face_recognition.face_encodings(
                        image_array,
                        face_locations
                    )

                    if len(encodings) != 1:
                        st.error(
                            "Could not encode the face. "
                            "Please try another photo."
                        )

                    else:
                        student = pending

                        if student is None:
                          st.error("Student details are missing. Please register the student again.")
                          st.stop()

                        student_id_value = student["student_id"]

                        photos_dir = Path("photos")
                        photos_dir.mkdir(exist_ok=True)

                        image_path = (
                            photos_dir / f"{student_id_value}.jpg"
                        )

                        student_added = database.add_student(
                            student_id=student_id_value,
                            name=student["name"],
                            department=student["department"],
                            semester=student["semester"],
                            face_image=str(image_path)
                        )

                        if not student_added:
                            st.error(
                                "Could not register the student. "
                                "The Student ID may already exist."
                            )

                        else:
                            try:
                                image.save(image_path, format="JPEG")

                                embedding_added = (
                                    database.add_face_embedding(
                                        student_id_value,
                                        encodings[0]
                                    )
                                )

                                if not embedding_added:
                                    st.error(
                                        "Student was saved, but the face "
                                        "encoding could not be saved. "
                                        "Please do not register this ID "
                                        "again; contact me to repair it."
                                    )

                                else:
                                    st.session_state.pop(
                                        "pending_student", None
                                    )
                                    st.success(
                                        f"{student['name']} registered "
                                        "with a face encoding!"
                                    )
                                    st.rerun()

                            except Exception as error:
                                st.error(
                                    f"An error occurred while saving: {error}"
                                )

    st.divider()
    st.subheader("📋 Student Database")

    students = database.get_students()

    if students:

        student_data = []

        for student in students:
            embeddings = database.get_face_embeddings(
                student["student_id"]
            )

            student_data.append({
                "Student ID": student["student_id"],
                "Student Name": student["name"],
                "Department": student["department"] or "-",
                "Semester": student["semester"] or "-",
                "Face Registered": "Yes" if embeddings else "No"
            })

        df = pd.DataFrame(student_data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(f"Total registered students: {len(students)}")

    else:
        st.info("No students registered yet.")

        
    st.markdown("---")
    st.subheader("🗑️ Delete Student")

    students_for_delete = database.get_students()

    if students_for_delete:
        student_options = {
            f"{student['name']} ({student['student_id']})":
            student["student_id"]
            for student in students_for_delete
        }

        selected_student = st.selectbox(
            "Select a student to delete",
            options=list(student_options.keys()),
            key="delete_student_select"
        )

        confirm_delete = st.checkbox(
            "I confirm that I want to delete this student",
            key="confirm_delete_student"
        )

        if st.button("🗑️ Delete Student", key="delete_student_button"):
            if confirm_delete:
                student_id_to_delete = student_options[selected_student]

                if database.delete_student(student_id_to_delete):
                    st.success("Student deleted successfully.")
                    st.rerun()
                else:
                    st.error("Student could not be deleted.")
            else:
                st.warning("Please confirm before deleting.")
    else:
        st.info("No registered students to delete.")



# =========================================================
# ATTENDANCE RECORDS
# =========================================================

elif page == "📋 Attendance Records":

    st.markdown(
        '<div class="section-title">'
        '📋 Attendance Records'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.attendance:

        df = pd.DataFrame(
            st.session_state.attendance
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Attendance CSV",
            csv,
            file_name=(
                f"FaceMark_Attendance_"
                f"{datetime.now().strftime('%Y-%m-%d')}.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.info(
            "No attendance records available."
        )