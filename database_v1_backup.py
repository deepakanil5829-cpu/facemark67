import sqlite3
from pathlib import Path
from datetime import datetime
import pickle
import numpy as np


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(exist_ok=True)

DATABASE_PATH = DATABASE_DIR / "facemark.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """Create and return a connection to the FaceMark database."""

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():
    """Create all required database tables."""

    connection = get_connection()
    cursor = connection.cursor()

    # --------------------------------------------------------
    # STUDENTS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT,
            semester TEXT,
            face_image TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # FACE EMBEDDINGS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,

            FOREIGN KEY (student_id)
                REFERENCES students(student_id)
                ON DELETE CASCADE
        )
    """)

    # --------------------------------------------------------
    # ATTENDANCE
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Present',
            confidence REAL,

            FOREIGN KEY (student_id)
                REFERENCES students(student_id),

            UNIQUE(student_id, date)
        )
    """)

    connection.commit()
    connection.close()


# ============================================================
# STUDENT FUNCTIONS
# ============================================================

def add_student(
    student_id,
    name,
    department="",
    semester="",
    face_image=""
):
    """Add a student to the database."""

    connection = get_connection()
    cursor = connection.cursor()

    try:

        cursor.execute("""
            INSERT INTO students (
                student_id,
                name,
                department,
                semester,
                face_image,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            student_id,
            name,
            department,
            semester,
            face_image,
            datetime.now().isoformat(timespec="seconds")
        ))

        connection.commit()

        return True

    except sqlite3.IntegrityError:

        return False

    finally:

        connection.close()


def get_students():
    """Return all registered students."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM students
        ORDER BY name
    """)

    students = cursor.fetchall()

    connection.close()

    return students


def get_student(student_id):
    """Find a student using their student ID."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM students
        WHERE student_id = ?
    """, (student_id,))

    student = cursor.fetchone()

    connection.close()

    return student


# ============================================================
# FACE EMBEDDING FUNCTIONS
# ============================================================

def add_face_embedding(student_id, embedding):
    """
    Store a face encoding for a student.

    face_recognition produces a NumPy array.
    We serialize it into a SQLite BLOB.
    """

    connection = get_connection()
    cursor = connection.cursor()

    try:

        # Convert NumPy array → bytes
        embedding_data = pickle.dumps(
            np.asarray(embedding)
        )

        cursor.execute("""
            INSERT INTO face_embeddings (
                student_id,
                embedding,
                created_at
            )
            VALUES (?, ?, ?)
        """, (
            student_id,
            embedding_data,
            datetime.now().isoformat(timespec="seconds")
        ))

        connection.commit()

        return True

    except sqlite3.Error:

        return False

    finally:

        connection.close()


def get_face_embeddings(student_id):
    """
    Return all face embeddings belonging to a student.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT embedding
        FROM face_embeddings
        WHERE student_id = ?
    """, (student_id,))

    rows = cursor.fetchall()

    connection.close()

    embeddings = []

    for row in rows:

        embedding = pickle.loads(
            row["embedding"]
        )

        embeddings.append(embedding)

    return embeddings


def get_all_face_embeddings():
    """
    Return all registered face embeddings.

    Returns:

        [
            {
                "student_id": "...",
                "embedding": numpy_array
            },
            ...
        ]
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            student_id,
            embedding
        FROM face_embeddings
    """)

    rows = cursor.fetchall()

    connection.close()

    embeddings = []

    for row in rows:

        embedding = pickle.loads(
            row["embedding"]
        )

        embeddings.append({
            "student_id": row["student_id"],
            "embedding": embedding
        })

    return embeddings


def delete_face_embeddings(student_id):
    """Delete all face embeddings for a student."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM face_embeddings
        WHERE student_id = ?
    """, (student_id,))

    connection.commit()
    connection.close()


# ============================================================
# ATTENDANCE FUNCTIONS
# ============================================================

def mark_attendance(student_id, confidence=None):
    """Mark a student present for today."""

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    connection = get_connection()
    cursor = connection.cursor()

    try:

        cursor.execute("""
            INSERT INTO attendance (
                student_id,
                date,
                time,
                status,
                confidence
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            student_id,
            date,
            time,
            "Present",
            confidence
        ))

        connection.commit()

        return True

    except sqlite3.IntegrityError:

        # Already marked today
        return False

    finally:

        connection.close()


def get_today_attendance():
    """Return today's attendance records."""

    today = datetime.now().strftime("%Y-%m-%d")

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            a.student_id,
            s.name,
            s.department,
            s.semester,
            a.date,
            a.time,
            a.status,
            a.confidence

        FROM attendance a

        JOIN students s
            ON a.student_id = s.student_id

        WHERE a.date = ?

        ORDER BY a.time
    """, (today,))

    records = cursor.fetchall()

    connection.close()

    return records


# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_database()


def delete_student(student_id):
    """Delete a student and their related records."""

    connection = get_connection()
    cursor = connection.cursor()

    try:
        # Get the student's photo path
        cursor.execute(
            "SELECT face_image FROM students WHERE student_id = ?",
            (student_id,)
        )
        student = cursor.fetchone()

        if student is None:
            return False

        # Delete related records first
        cursor.execute(
            "DELETE FROM attendance WHERE student_id = ?",
            (student_id,)
        )

        cursor.execute(
            "DELETE FROM face_embeddings WHERE student_id = ?",
            (student_id,)
        )

        # Delete the student
        cursor.execute(
            "DELETE FROM students WHERE student_id = ?",
            (student_id,)
        )

        connection.commit()

        # Remove the student's photo if it exists
        if student["face_image"]:
            from pathlib import Path

            photo_path = Path(student["face_image"])

            if not photo_path.is_absolute():
                photo_path = BASE_DIR / photo_path

            try:
                if photo_path.exists():
                    photo_path.unlink()
            except OSError:
                pass

        return True

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


