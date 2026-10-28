import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
import database
import time
from datetime import datetime


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FaceMark | Smart Attendance",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= FACE MARK UI THEME =================


# ================= PREMIUM FACEMARK THEME =================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0B1020;
    --sidebar: #0E1528;
    --panel: #131D33;
    --panel-hover: #192743;
    --blue: #6C7CFF;
    --cyan: #42D6E8;
    --green: #39DCA7;
    --text: #F4F7FF;
    --muted: #98A6C3;
    --border: #273450;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(11, 16, 32, 0.95);
}

[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #C4CEE2;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--text) !important;
    letter-spacing: -0.5px;
    font-weight: 700 !important;
}

p, label, .stCaption {
    color: #B5C1D7;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #172541, #111A2D);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 800;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    background: #0D162A;
    color: white;
    border: 1px solid #30405F;
    border-radius: 11px;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #0D162A;
    color: white;
    border-color: #30405F;
    border-radius: 11px;
}

.stButton > button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
    background: linear-gradient(110deg, #586BFA, #7884FF);
    color: white;
    border: 1px solid #7784FF;
    border-radius: 11px;
    min-height: 44px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
    background: #4558E8;
    border-color: #A5AEFF;
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px #586BFA30;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
}

[data-testid="stExpander"],
[data-testid="stForm"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
}

[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--border);
}

hr {
    border-color: var(--border);
}

.facemark-title {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1.2px;
    color: var(--text);
    margin-bottom: 4px;
}

.facemark-subtitle {
    font-size: 14px;
    color: var(--muted);
    margin-bottom: 28px;
}

.section-title {
    color: var(--text);
    font-size: 25px;
    font-weight: 750;
    letter-spacing: -0.7px;
    margin: 8px 0 22px;
}

.dashboard-card {
    background: linear-gradient(145deg, #172541, #111A2D);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px;
    min-height: 135px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.dashboard-card:hover {
    border-color: #6878E8;
    transform: translateY(-2px);
}

.card-number {
    color: var(--text);
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1px;
    overflow-wrap: anywhere;
}

.card-title {
    color: var(--muted);
    font-size: 13px;
    margin-top: 10px;
    font-weight: 500;
}

#MainMenu, footer {
    visibility: hidden;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .dashboard-card {
        padding: 16px;
        min-height: 110px;
    }

    .card-number {
        font-size: 25px;
    }

    .facemark-title {
        font-size: 28px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------- FACEMARK LOADING SCREEN ----------
if "facemark_splash_shown" not in st.session_state:
    splash = st.empty()

    splash.markdown(
        """
        <style>
        .fm-splash {
            position: fixed;
            inset: 0;
            z-index: 999999;
            background: #050505;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-family: Arial, sans-serif;
            animation: fmFadeIn 0.5s ease;
        }

        .fm-logo {
            width: 76px;
            height: 76px;
            border: 2px solid #ffffff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 36px;
            font-weight: 800;
            margin-bottom: 25px;
            animation: fmPulse 1.2s ease-in-out infinite;
        }

        .fm-brand {
            font-size: clamp(30px, 5vw, 48px);
            font-weight: 900;
            letter-spacing: 7px;
            margin-left: 7px;
        }

        .fm-tagline {
            color: #999999;
            font-size: 10px;
            letter-spacing: 4px;
            margin-top: 12px;
        }

        .fm-loader {
            width: 190px;
            height: 3px;
            background: #292929;
            margin-top: 48px;
            overflow: hidden;
        }

        .fm-loader-bar {
            width: 40%;
            height: 100%;
            background: #ffffff;
            animation: fmLoading 1.1s ease-in-out infinite;
        }

        .fm-status {
            margin-top: 18px;
            color: #aaaaaa;
            font-size: 10px;
            letter-spacing: 2px;
        }

        @keyframes fmLoading {
            from { transform: translateX(-110%); }
            to   { transform: translateX(360%); }
        }

        @keyframes fmPulse {
            0%, 100% { transform: scale(1); }
            50%      { transform: scale(1.07); }
        }

        @keyframes fmFadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        </style>

        <div class="fm-splash">
            <div class="fm-logo">F</div>
            <div class="fm-brand">FACEMARK</div>
            <div class="fm-tagline">
                AI-POWERED ATTENDANCE
            </div>
            <div class="fm-loader">
                <div class="fm-loader-bar"></div>
            </div>
            <div class="fm-status">
                INITIALIZING FACE RECOGNITION...
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    time.sleep(2.2)
    splash.empty()
    st.session_state["facemark_splash_shown"] = True

# ---------- END LOADING SCREEN ----------




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
# FACEMARK — PUMA-INSPIRED UI
# =========================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #101010;
    color: #ffffff;
}

/* Hide default Streamlit chrome */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080808;
    border-right: 1px solid #292929;
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}

/* Brand */
.fm-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 5px 0 25px 0;
}

.fm-logo {
    width: 48px;
    height: 48px;
    background: #d7ff00;
    color: #080808;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 27px;
    font-weight: 1000;
    border-radius: 8px;
}

.fm-brand-name {
    color: #ffffff;
    font-size: 23px;
    font-weight: 950;
    letter-spacing: -1px;
    line-height: 1.1;
}

.fm-brand-subtitle {
    color: #8c8c8c;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-top: 5px;
}

/* Section label */
.fm-section {
    color: #777777;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
    margin: 26px 0 12px 8px;
}

/* Navigation buttons */
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 52px;
    background: transparent;
    color: #d4d4d4;
    border: 1px solid transparent;
    border-radius: 7px;
    text-align: left;
    justify-content: flex-start;
    padding: 0 15px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.5px;
    transition: all 0.18s ease;
}

/* Hover */
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #1b1b1b;
    color: #d7ff00;
    border-color: #333333;
}

/* Main content */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 2rem;
}

/* Footer */
.fm-footer {
    margin-top: 35px;
    padding-top: 20px;
    border-top: 1px solid #292929;
}

.fm-status {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #bcbcbc;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.fm-status-dot {
    height: 8px;
    width: 8px;
    background: #d7ff00;
    border-radius: 50%;
}

.fm-footer-note {
    color: #666666;
    font-size: 9px;
    margin-top: 12px;
    letter-spacing: 1px;
}

/* Main page headings */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 900 !important;
    letter-spacing: -0.7px;
}

/* Lime accent */
.fm-accent {
    color: #d7ff00;
    font-weight: 900;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

# =========================================================
# PUMA-INSPIRED NAVIGATION STYLE
# =========================================================


# =========================================================
# SIDEBAR
# =========================================================


# =========================================================
# FACEMARK NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="fm-brand">
        <div class="fm-logo">F</div>
        <div>
            <div class="fm-brand-name">FACEMARK</div>
            <div class="fm-brand-subtitle">AI ATTENDANCE SYSTEM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="fm-section">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    # Initialize selected page
    if "fm_page" not in st.session_state:
        st.session_state.fm_page = "Dashboard"

    # Custom navigation buttons
    navigation = [
        ("Dashboard", "▦", "DASHBOARD"),
        ("Attendance", "◎", "MARK ATTENDANCE"),
        ("Students", "♙", "STUDENTS"),
        ("Attendance Records", "▤", "ATTENDANCE RECORDS"),
    ]

    for page_name, symbol, label in navigation:

        selected = st.session_state.fm_page == page_name

        if selected:
            button_label = f"🟩  {symbol}   {label}"
        else:
            button_label = f"{symbol}   {label}"

        if st.button(
            button_label,
            key=f"fm_nav_{page_name}",
            use_container_width=True
        ):
            st.session_state.fm_page = page_name
            st.rerun()

    # System footer
    st.markdown("""
    <div class="fm-footer">
        <div class="fm-status">
            <span class="fm-status-dot"></span>
            SYSTEM ONLINE
        </div>
        <div class="fm-footer-note">
            FACEMARK / V1.0<br>
            COMPUTER VISION PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

# Keep this variable before your existing dashboard conditions
page = st.session_state.fm_page

page_mapping = {
    "Dashboard": "📊 Dashboard",
    "Attendance": "📷 Attendance",
    "Students": "👨‍🎓 Students",
    "Attendance Records": "📋 Attendance Records",
}

page = page_mapping[st.session_state.fm_page]





# =========================================================
# HEADER
# =========================================================


# ================= PREMIUM APPLICATION HEADER =================

header_col, status_col = st.columns([4, 1])

with header_col:
    st.markdown(
        '<div class="facemark-title">◈ FaceMark</div>'
        '<div class="facemark-subtitle">'
        'Intelligent attendance management · AI-powered recognition'
        '</div>',
        unsafe_allow_html=True
    )

with status_col:
    st.markdown(
        """
        <div style="
            text-align:right;
            padding-top:10px;
            color:#39DCA7;
            font-size:13px;
            font-weight:600;
        ">● SYSTEM READY</div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DASHBOARD
# =========================================================



if page == "📊 Dashboard":

    total_students = len(database.get_students())
    present_students = len(st.session_state.attendance)
    current_date = datetime.now().strftime("%d %B %Y")

    st.markdown("""
    <style>
    .fm-dash-hero {
        position: relative;
        overflow: hidden;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #292846;
        background:
            radial-gradient(ellipse at 90% 10%,
                #5146a833, transparent 42%),
            linear-gradient(125deg, #11121f, #080910);
        margin-bottom: 24px;
    }

    .fm-dash-eyebrow {
        color: #a49aff;
        font-size: 10px;
        letter-spacing: 3px;
        font-weight: 800;
        margin-bottom: 13px;
    }

    .fm-dash-heading {
        color: #f7f7ff;
        font-size: clamp(27px, 4vw, 40px);
        font-weight: 900;
        letter-spacing: -1.4px;
        margin: 0;
    }

    .fm-dash-gradient {
        background: linear-gradient(90deg, #8278ff, #c084fc);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .fm-dash-description {
        color: #989bb3;
        font-size: 13px;
        line-height: 1.8;
        margin-top: 12px;
    }

    .fm-dash-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        margin-top: 20px;
        border-radius: 30px;
        border: 1px solid #37335c;
        background: #19172b;
        color: #c4bcff;
        font-size: 10px;
        letter-spacing: 1px;
    }

    .fm-dash-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #9b8cff;
        box-shadow: 0 0 10px #8278ff;
    }

    .fm-dash-section {
        color: #f1f0ff;
        font-size: 17px;
        font-weight: 800;
        margin: 26px 0 5px;
    }

    .fm-dash-muted {
        color: #8589a3;
        font-size: 12px;
        margin-bottom: 16px;
    }

    .fm-stat-card {
        min-height: 125px;
        padding: 19px;
        border: 1px solid #292a40;
        border-radius: 16px;
        background: linear-gradient(145deg, #151625, #0c0d16);
        box-shadow: 0 8px 24px #00000015;
    }

    .fm-stat-top {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #a5a8c0;
        font-size: 11px;
        font-weight: 600;
    }

    .fm-stat-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 9px;
        background: #5146a82b;
        color: #aaa0ff;
        font-size: 16px;
    }

    .fm-stat-value {
        color: #faf9ff;
        font-size: clamp(22px, 2.5vw, 31px);
        font-weight: 900;
        letter-spacing: -1px;
        margin-top: 15px;
        overflow-wrap: anywhere;
    }

    .fm-stat-caption {
        color: #777c99;
        font-size: 10px;
        margin-top: 5px;
    }

    .fm-step-card {
        min-height: 148px;
        padding: 18px;
        border: 1px solid #282940;
        border-radius: 15px;
        background: linear-gradient(150deg, #141523, #0b0c14);
    }

    .fm-step-number {
        color: #9387ff;
        font-size: 10px;
        letter-spacing: 2px;
        font-weight: 800;
    }

    .fm-step-title {
        color: #f0efff;
        font-size: 14px;
        font-weight: 800;
        margin-top: 17px;
    }

    .fm-step-description {
        color: #8c90aa;
        font-size: 11px;
        line-height: 1.7;
        margin-top: 8px;
    }

    .fm-dash-footer {
        text-align: center;
        color: #666a84;
        font-size: 9px;
        letter-spacing: 2px;
        padding: 22px 0 5px;
    }
    </style>

    <div class="fm-dash-hero">
        <div class="fm-dash-eyebrow">
            FACEMARK / CONTROL CENTER
        </div>
        <h1 class="fm-dash-heading">
            Attendance,<br>
            <span class="fm-dash-gradient">Reimagined.</span>
        </h1>
        <div class="fm-dash-description">
            Welcome to your intelligent attendance workspace.<br>
            Manage student registrations and monitor attendance
            through AI-powered face recognition.
        </div>
        <div class="fm-dash-status">
            <span class="fm-dash-dot"></span>
            SYSTEM DASHBOARD
        </div>
    </div>

    <div class="fm-dash-section">Overview</div>
    <div class="fm-dash-muted">
        A quick snapshot of your attendance system
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="fm-stat-card">
            <div class="fm-stat-top">
                <span class="fm-stat-icon">♙</span>
                REGISTERED STUDENTS
            </div>
            <div class="fm-stat-value">{total_students}</div>
            <div class="fm-stat-caption">
                Students in the database
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="fm-stat-card">
            <div class="fm-stat-top">
                <span class="fm-stat-icon">✓</span>
                PRESENT TODAY
            </div>
            <div class="fm-stat-value">{present_students}</div>
            <div class="fm-stat-caption">
                Recorded in this session
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="fm-stat-card">
            <div class="fm-stat-top">
                <span class="fm-stat-icon">◷</span>
                CURRENT DATE
            </div>
            <div class="fm-stat-value"
                 style="font-size:clamp(16px, 2vw, 23px);">
                {current_date}
            </div>
            <div class="fm-stat-caption">
                Local system date
            </div>
        </div>
        """, unsafe_allow_html=True)



    st.markdown("""
    <div class="fm-dash-footer">
        FACEMARK &nbsp; / &nbsp;
        AI-POWERED ATTENDANCE SYSTEM &nbsp; / &nbsp; V1.0
    </div>
    """, unsafe_allow_html=True)


   

# =========================================================
# ATTENDANCE
# =========================================================

elif page == "📷 Attendance":

    
    st.markdown("""
    <div class="facemark-title">Attendance Studio</div>
    <div class="facemark-subtitle">
        Real-time face recognition and attendance monitoring
    </div>

    <div style="
        background: linear-gradient(120deg, #152440, #111A2D);
        border: 1px solid #2B3B5B;
        border-radius: 15px;
        padding: 17px 20px;
        margin-bottom: 18px;
    ">
        <div style="font-size:15px;font-weight:700;color:#F4F7FF;">
            ◉ Live recognition workspace
        </div>
        <div style="font-size:13px;color:#98A6C3;margin-top:6px;">
            Allow camera access, face the camera, and wait for
            FaceMark to identify registered students.
        </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("🚀 Start Attendance", use_container_width=True):
            st.session_state.camera_started = True

    with col2:
        if st.button("🔄 Reset", use_container_width=True):
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

    
    fm_page_header(
        "Student Management",
        "Register students and manage their face recognition profiles"
    )

    st.markdown(
        """
        <div style="
            background: linear-gradient(120deg, #152440, #111A2D);
            border: 1px solid #2B3B5B;
            border-radius: 15px;
            padding: 17px 20px;
            margin-bottom: 20px;
        ">
            <div style="font-size:15px;font-weight:700;color:#F4F7FF;">
                👨‍🎓 Student Directory
            </div>
            <div style="font-size:13px;color:#98A6C3;margin-top:6px;">
                Add student details, register face images, and manage
                existing student records.
            </div>
        </div>
        """,
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

    
    fm_page_header(
        "Attendance Records",
        "Review, manage, and export attendance for your session"
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