import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime
from PIL import Image
import pandas as pd
import threading
import shutil

# Configuration
KNOWN_FACES_DIR = "students_faces"
REPORTS_DIR = "session_reports" # <-- NEW: Directory to save session reports
TOLERANCE = 0.5
FRAME_THICKNESS = 2
FONT_THICKNESS = 1
RESIZE_SCALE = 0.25
SESSION_DURATION = 45 * 60  # 45 minutes in seconds

# Create necessary directories if they don't exist
for dir_path in [KNOWN_FACES_DIR, REPORTS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

# Initialize Session State
def init_session_state():
    if "tracker" not in st.session_state:
        st.session_state.tracker = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "last_frame" not in st.session_state:
        st.session_state.last_frame = None
    if "csv_data" not in st.session_state:
        st.session_state.csv_data = None
    if "known_students" not in st.session_state:
        st.session_state.known_students = {}
    if "session_start" not in st.session_state:
        st.session_state.session_start = None
    if "session_timer" not in st.session_state:
        st.session_state.session_timer = None
    if "remaining_time" not in st.session_state:
        st.session_state.remaining_time = SESSION_DURATION
    if "show_registration_form" not in st.session_state:
        st.session_state.show_registration_form = False
    # <-- NEW: State for attendance history page
    if "show_history" not in st.session_state:
        st.session_state.show_history = False
    if "show_student_management" not in st.session_state:
        st.session_state.show_student_management = False

# Load Known Students (cached for performance)
@st.cache_data
def load_known_faces():
    st.session_state.known_students = {}
    for folder in os.listdir(KNOWN_FACES_DIR):
        folder_path = os.path.join(KNOWN_FACES_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        parts = folder.split("_")
        student_id = parts[0]
        name = " ".join(parts[1:]) if len(parts) > 1 else student_id
        
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder_path, img_file)
                try:
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        if student_id not in st.session_state.known_students:
                            st.session_state.known_students[student_id] = {"name": name, "encodings": []}
                        st.session_state.known_students[student_id]["encodings"].append(encodings[0])
                except Exception as e:
                    st.error(f"Error processing {img_file} in {folder}: {e}")
    
    # This message now appears in the sidebar after loading
    if 'known_students' in st.session_state and st.session_state.known_students:
        st.sidebar.success(f"Loaded {len(st.session_state.known_students)} students.")

# StudentTracker Class (handles the logic for tracking presence)
class StudentTracker:
    # ... (No changes needed in this class)
    def __init__(self):
        self.students = {}
        for student_id, student_data in st.session_state.known_students.items():
            self.students[student_id] = {
                "name": student_data["name"],
                "in_frame": False,
                "start_time": None,
                "total_time": 0.0,
                "first_seen": None,
                "last_seen": None,
                "time_in": None,
                "time_out": None,
            }
    
    def update_presence(self, student_id, in_frame, current_time):
        student = self.students.get(student_id)
        if not student: return
        
        if in_frame and not student["in_frame"]:
            student["in_frame"] = True
            student["start_time"] = current_time
            if student["first_seen"] is None:
                student["first_seen"] = current_time
                student["time_in"] = datetime.fromtimestamp(current_time).strftime("%H:%M:%S")

        elif not in_frame and student["in_frame"]:
            student["in_frame"] = False
            if student["start_time"]:
                student["total_time"] += current_time - student["start_time"]
                student["start_time"] = None
                student["time_out"] = datetime.fromtimestamp(current_time).strftime("%H:%M:%S")
    
    def final_update(self, current_time):
        for data in self.students.values():
            if data["in_frame"] and data["start_time"]:
                data["total_time"] += current_time - data["start_time"]
                data["in_frame"] = False
                data["start_time"] = None
                data["time_out"] = datetime.fromtimestamp(current_time).strftime("%H:%M:%S")
    
    def get_csv_data(self):
        session_start = st.session_state.session_start
        session_end = time.time()
        session_start_dt = datetime.fromtimestamp(session_start)
        session_date = session_start_dt.strftime("%Y-%m-%d")
        session_start_str = session_start_dt.strftime("%H:%M:%S")
        session_end_str = datetime.fromtimestamp(session_end).strftime("%H:%M:%S")
        session_duration_secs = session_end - session_start
        data = []

        for student_id, info in self.students.items():
            total_time_secs = round(info.get("total_time", 0.0), 2)
            total_time_mins = round(total_time_secs / 60, 2)
            
            if total_time_secs == 0:
                performance = "Absent"
                status = "Absent"
            else:
                time_ratio = total_time_secs / session_duration_secs if session_duration_secs > 0 else 0
                if time_ratio >= 0.75: performance = "Excellent"
                elif time_ratio >= 0.50: performance = "Very Good"
                elif time_ratio >= 0.25: performance = "Good"
                else: performance = "Poor"
                status = "Present"
            
            data.append({
                "Student ID": student_id, "Name": info["name"],
                "Session Start Time": session_start_str, "Session End Time": session_end_str,
                "Student Class Entering Time": info.get("time_in", "N/A"),
                "Student Last Seen Time": info.get("last_seen", "N/A"),
                "Student Check Out Time": info.get("time_out", "N/A"),
                "Total Time (seconds)": total_time_secs, "Total Time (minutes)": total_time_mins,
                "Performance": performance, "Status": status, "Session Date": session_date,
            })
        return data

# Session timer thread
def session_timer():
    # ... (No changes needed in this function)
    try:
        while st.session_state.get("is_running", False) and st.session_state.get("remaining_time", 0) > 0:
            time.sleep(1)
            st.session_state.remaining_time -= 1
        if st.session_state.get("remaining_time", 1) <= 0:
            st.session_state.is_running = False
            if st.session_state.tracker:
                current_time = time.time()
                st.session_state.tracker.final_update(current_time)
                st.session_state.csv_data = st.session_state.tracker.get_csv_data()
            if st.session_state.cap and st.session_state.cap.isOpened():
                st.session_state.cap.release()
            st.rerun()
    except Exception as e:
        print(f"Error in session timer thread: {e}")

# <-- MODIFIED: Student registration form now uses file upload
def registration_form():
    st.subheader("Register New Student")
    with st.form(key="registration_form"):
        student_id = st.text_input("Student ID (must be unique)")
        student_name = st.text_input("Full Name")
        contact_no = st.text_input("Contact Number (Optional)")
        
        # <-- MODIFIED: Replaced camera_input with file_uploader
        uploaded_photo = st.file_uploader(
            "Upload Student Photo (one image only)", 
            type=["jpg", "jpeg", "png"]
        )
        submitted = st.form_submit_button("Register Student")

        if submitted:
            if not student_id or not student_name or not uploaded_photo:
                st.error("Please provide Student ID, Name, and upload a photo.")
                return

            folder_name = f"{student_id}_{student_name.replace(' ', '_')}"
            student_dir = os.path.join(KNOWN_FACES_DIR, folder_name)
            os.makedirs(student_dir, exist_ok=True)

            try:
                img = Image.open(uploaded_photo)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # Save as 1.jpg, overwriting if it exists, since we only want one photo.
                save_path = os.path.join(student_dir, "1.jpg")
                img.save(save_path, "JPEG")

                info_path = os.path.join(student_dir, "info.txt")
                with open(info_path, "w") as f:
                    f.write(f"ID: {student_id}\nName: {student_name}\nContact: {contact_no}\n")
                
                st.success(f"Student '{student_name}' registered successfully!")
                load_known_faces.clear()
                st.session_state.show_registration_form = False
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save image: {e}")

# <-- NEW: Function to display attendance history
def display_attendance_history():
    st.header("📜 Attendance History")
    
    report_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.csv')]
    
    if not report_files:
        st.warning("No past session reports found.")
        return

    all_reports_df = pd.DataFrame()
    try:
        df_list = [pd.read_csv(os.path.join(REPORTS_DIR, f)) for f in report_files]
        if df_list:
            all_reports_df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        st.error(f"Error loading report files: {e}")
        return

    if all_reports_df.empty:
        st.info("No attendance data to show yet.")
        return
        
    st.dataframe(all_reports_df)
    
    # Allow downloading the consolidated report
    csv_data = all_reports_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full History as CSV",
        data=csv_data,
        file_name="full_attendance_history.csv",
        mime="text/csv"
    )

# Main Application
def main():
    init_session_state()
    st.set_page_config(page_title="Classroom Tracker", page_icon="👨‍🎓", layout="wide")

def apply_custom_css():
     """Applies custom CSS to improve the application's visual style."""
     st.markdown("""
    <style>
        /* Main container and sidebar styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background-color: #111;
        }
        /* Style the metric widget in the sidebar */
        [data-testid="stMetric"] {
            background-color: #222;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        /* Change the color of the metric value */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem;
            color: #00A8E8; /* A vibrant blue */
        }
    </style>
    """, unsafe_allow_html=True)

# In your main() function:
def main():
    init_session_state()
    st.set_page_config(page_title="Classroom Tracker", page_icon="👨‍🎓", layout="wide")
    
    # ADD THIS LINE TO CALL THE FUNCTION
    apply_custom_css()

    st.markdown("<h1>👨‍🎓 Student Tracking System</h1>", unsafe_allow_html=True)
    # ... rest of your main function

    #st.markdown("<h1>👨‍🎓 Student Tracking System</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # <-- MODIFIED: Logo re-enabled. Change the path to your actual logo file.
        try:
            st.image("D:\Student_time_tracking\logo.jpg", use_container_width=True , width= 'stretch')
        except Exception:
            st.info("Update the path in the code to display your college logo.")

        # st.header("Session Controls")
        # if st.session_state.is_running:
        #     mins, secs = divmod(st.session_state.remaining_time, 60)
        #     st.metric(label="Time Remaining", value=f"{mins:02d}:{secs:02d}")
        # else:
        #     st.metric(label="Session Duration", value=f"{SESSION_DURATION//60} minutes")
        # st.divider()

        # --- START OF NEW CODE TO ADD ---
        st.header("Session Controls")
        
        # Determine if a session is running to disable widgets
        is_running = st.session_state.is_running

        # Interactive widget for session duration
        session_duration_minutes = st.number_input(
            "Session Duration (minutes)", 
            min_value=1, 
            max_value=180, 
            value=45, 
            disabled=is_running
        )
        SESSION_DURATION = session_duration_minutes * 60 # Convert to seconds
        
        # Interactive widget for face recognition tolerance
        TOLERANCE = st.slider(
            "Face Recognition Tolerance", 
            min_value=0.30, 
            max_value=0.70, 
            value=0.50, # A lower value is stricter
            step=0.05,
            disabled=is_running,
            help="Lower values make face matching more strict. 0.5 is a good balance."
        )

        # Display the timer metric
        if is_running:
            mins, secs = divmod(st.session_state.remaining_time, 60)
            st.metric(label="Time Remaining", value=f"{mins:02d}:{secs:02d}")
        else:
            st.metric(label="Session Duration", value=f"{session_duration_minutes} minutes")

        st.divider()
# --- END OF NEW CODE TO ADD ---

        # st.subheader("Student Management")
        # if st.button("Register New Student", use_container_width=True):
        #     st.session_state.show_registration_form = True
        #     st.session_state.show_history = False # Hide other pages
        #     st.rerun()
        
        # if st.button("Manage Students", use_container_width=True):
        #     st.session_state.show_student_management = True
        #     st.session_state.show_registration_form = False # Hide other pages
        #     st.session_state.show_history = False # Hide other pages
        #     st.rerun()

        # # <-- NEW: Attendance History Button
        # if st.button("Attendance History", use_container_width=True):
        #     st.session_state.show_history = True
        #     st.session_state.show_registration_form = False # Hide other pages
        #     st.rerun()

        
        if st.button("Reload Students List", use_container_width=True):
            load_known_faces.clear()
        
        load_known_faces() # Load faces and display count in sidebar
        st.divider()

        st.subheader("Session Control")
        if not st.session_state.is_running:
            if st.button("Start Classroom Session", use_container_width=True, disabled=not st.session_state.known_students):
                st.session_state.tracker = StudentTracker()
                st.session_state.cap = cv2.VideoCapture(0)
                st.session_state.is_running = True
                st.session_state.show_registration_form = False
                st.session_state.show_history = False
                st.session_state.session_start = time.time()
                st.session_state.remaining_time = SESSION_DURATION
                st.session_state.session_timer = threading.Thread(target=session_timer)
                st.session_state.session_timer.start()
                st.rerun()
        else:
            if st.button("End Session Now", use_container_width=True, type="primary"):
                st.session_state.is_running = False
                if st.session_state.tracker:
                    current_time = time.time()
                    st.session_state.tracker.final_update(current_time)
                    st.session_state.csv_data = st.session_state.tracker.get_csv_data()
                if st.session_state.cap and st.session_state.cap.isOpened():
                    st.session_state.cap.release()
                st.rerun()

    # Conditional page display
    if st.session_state.show_registration_form:
        registration_form()
        if st.button("Back to Main Page"):
            st.session_state.show_registration_form = False
            st.rerun()

    elif st.session_state.show_student_management:
        display_student_management_page(KNOWN_FACES_DIR)
        if st.button("Back to Main Page"):
            st.session_state.show_student_management = False
            st.rerun()

    elif st.session_state.show_history:
        display_attendance_history()
        if st.button("Back to Main Page"):
            st.session_state.show_history = False
            st.rerun()
    else:
        display_main_tracker(TOLERANCE)

def display_live_dashboard(session_start_time):
        """
        This function displays a live dashboard of student attendance during a session.
        It calculates and shows the attendance percentage for each student in real-time.
        
        Args:
            session_start_time (float): The timestamp when the session began.
        """
        st.info("Tracking in progress...")
        st.subheader("Live Attendance Dashboard")

        if hasattr(st.session_state, "tracker") and st.session_state.tracker:
            session_elapsed = time.time() - session_start_time

        for student_id, data in st.session_state.tracker.students.items():
            current_total_time = data["total_time"]
            
            if data["in_frame"] and data["start_time"]:
                current_total_time += (time.time() - data["start_time"])

            attendance_percentage = (current_total_time / session_elapsed) * 100 if session_elapsed > 0 else 0
            attendance_percentage = min(100, attendance_percentage)

            status_text = "✅ In Frame" if data["in_frame"] else "❌ Not in Frame"
            
            with st.container():
                st.write(f"**{data['name']} ({student_id})**")
                st.progress(int(attendance_percentage))
                st.caption(f"{status_text} | Total time present: {current_total_time / 60:.2f} mins")
            st.markdown("---")



def display_student_management_page(known_faces_dir):
    """
    Creates a page to view, manage, and delete student profiles.
    This function reads student data directly from their respective folders.
    """
    st.header("🧑‍🎓 Student Profile Management")

    # Get a list of all directories inside the main student faces folder
    student_folders = [d for d in os.listdir(known_faces_dir) if os.path.isdir(os.path.join(known_faces_dir, d))]

    if not student_folders:
        st.warning("No students are registered yet. Register a new student to manage profiles.")
        return

    # Loop through each student folder found
    for folder_name in sorted(student_folders):
        student_dir = os.path.join(known_faces_dir, folder_name)
        
        # Extract ID and Name from the folder name (e.g., "001_Yoshita")
        parts = folder_name.split("_")
        student_id = parts[0]
        student_name = " ".join(parts[1:]) if len(parts) > 1 else student_id

        # Use st.container with a border to visually group each student's profile
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Find and display the student's photo
                image_path = os.path.join(student_dir, "1.jpg")
                if os.path.exists(image_path):
                    st.image(image_path, width=120)
                else:
                    st.caption("No photo found")

            with col2:
                st.subheader(f"{student_name}")
                st.write(f"**Student ID:** {student_id}")

                # Create a delete button. The 'key' is crucial here!
                # It must be unique for each button inside a loop.
                if st.button("Delete Profile", key=f"delete_{student_id}", type="primary"):
                    try:
                        # Use shutil.rmtree to delete the entire student folder and its contents
                        shutil.rmtree(student_dir)
                        st.success(f"Successfully deleted the profile for {student_name}.")
                        
                        # Clear the cached face data and rerun the app to reflect the change
                        load_known_faces.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting profile: {e}")

# def display_main_tracker(tolerance):
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         st.subheader("Classroom Camera Feed")
#         frame_placeholder = st.empty()
#         status_text = st.empty()

#         if st.session_state.is_running:
#             # Camera feed logic... (No changes here)
#             status_text.info("Live camera feed is active.")
#             while st.session_state.is_running and st.session_state.cap and st.session_state.cap.isOpened():
#                 ret, frame = st.session_state.cap.read()
#                 if not ret:
#                     status_text.error("Failed to capture video feed.")
#                     st.session_state.is_running = False
#                     break
#                 st.session_state.last_frame = frame.copy()
#                 small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
#                 rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#                 face_locations = face_recognition.face_locations(rgb_small_frame)
#                 face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
#                 if st.session_state.known_students:
#                     current_presence = {sid: False for sid in st.session_state.tracker.students}
#                     for encoding, location in zip(face_encodings, face_locations):
#                         best_match_id, best_match_name = "Unknown", "Unknown"
#                         best_distance = 1.0
#                         for sid, sdata in st.session_state.known_students.items():
#                             distances = face_recognition.face_distance(sdata["encodings"], encoding)
#                             min_dist = np.min(distances) if distances.size > 0 else 1.0
#                             # if min_dist < TOLERANCE and min_dist < best_distance:
#                             if min_dist < tolerance and min_dist < best_distance:
#                                 best_distance, best_match_id, best_match_name = min_dist, sid, sdata["name"]
#                         color = (0, 255, 0) if best_match_id != "Unknown" else (0, 0, 255)
#                         if best_match_id != "Unknown": current_presence[best_match_id] = True
#                         top, right, bottom, left = [int(v / RESIZE_SCALE) for v in location]
#                         cv2.rectangle(frame, (left, top), (right, bottom), color, FRAME_THICKNESS)
#                         label = f"{best_match_name}" + (f" ({best_match_id})" if best_match_id != "Unknown" else "")
#                         cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, FONT_THICKNESS)
#                     current_time = time.time()
#                     for sid in st.session_state.tracker.students:
#                         st.session_state.tracker.update_presence(sid, current_presence.get(sid, False), current_time)
#                 mins, secs = divmod(st.session_state.remaining_time, 60)
#                 timer_text = f"Session Time: {mins:02d}:{secs:02d}"
#                 cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#                 frame_placeholder.image(frame, channels="BGR")
#         else:
#             # Logic for when camera is off
#             if st.session_state.last_frame is not None:
#                 frame_placeholder.image(st.session_state.last_frame, channels="BGR")
#                 status_text.info("Session ended. Displaying last captured frame.")
#             else:
#                 frame_placeholder.image(Image.new("RGB", (640, 480), (50, 50, 50)), caption="Camera is off")
#                 status_text.info("Start a session to begin tracking.")
    
#     with col2:
#         st.subheader("Session Report")
#         if st.session_state.is_running:
#             # Live tracking display
#             # st.info("Tracking in progress...")
#             # for student_id, data in st.session_state.tracker.students.items():
#             #     total_time_min = data["total_time"] / 60
#             #     if data["in_frame"] and data["start_time"]:
#             #         total_time_min += (time.time() - data["start_time"]) / 60
#             #     status_icon = "✅" if data["in_frame"] else "❌"
#             #     st.write(f"**{data['name']}**: {status_icon} ({total_time_min:.2f} mins)")

#             # This one line calls our new dashboard function.
#             # We pass `st.session_state.session_start` so the dashboard
#             # knows the session's start time to calculate percentages.
#             display_live_dashboard(st.session_state.session_start)
#         elif st.session_state.csv_data:
#             # Post-session report display
#             st.success("Session completed!")
#             df = pd.DataFrame(st.session_state.csv_data)
#             st.dataframe(df)
            
#             # <-- MODIFIED: Auto-save the report and provide download
#             csv_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_classroomReport.csv"
#             csv_data = df.to_csv(index=False).encode("utf-8")

#             # Auto-save to reports directory
#             save_path = os.path.join(REPORTS_DIR, csv_filename)
#             with open(save_path, "wb") as f:
#                 f.write(csv_data)

#             st.download_button("Download Session Report", csv_data, csv_filename, "text/csv", use_container_width=True)
#             if st.button("Clear Session Data", use_container_width=True):
#                 st.session_state.csv_data = None
#                 st.session_state.last_frame = None
#                 st.rerun()
#         else:
#             # <-- MODIFIED: Removed the registered students list
#             st.info("The session report will appear here after the session ends.")
#             if not st.session_state.known_students:
#                 st.warning("No new students registered. Please register any left students & start a session." \
#                 " If already registered , do reload students list")


def display_main_tracker(tolerance):
    """
    Displays the main tracking interface, including the new control panel,
    camera feed, and session report sections.
    """

    # ----------------------------------------------------------------------
    # NEW: Control Panel / Dashboard Section
    # ----------------------------------------------------------------------
    st.subheader("Control Panel")

    # Display key metrics using columns
    total_students = len(st.session_state.known_students)
    status = "Active" if st.session_state.is_running else "Idle"

    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        st.metric(label="✅ Registered Students", value=total_students)
    with metric2:
        st.metric(label="🖥️ Session Status", value=status)
    with metric3:
        # A simple placeholder metric, can be expanded later
        st.metric(label="🕒 Last Session", value="N/A")

    st.markdown("---") # Visual separator

    # Display management buttons with icons in columns
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("➕ Register Student", use_container_width=True):
            st.session_state.show_registration_form = True
            st.rerun()
    with b2:
        if st.button("⚙️ Manage Students", use_container_width=True):
            st.session_state.show_student_management = True
            st.rerun()
    with b3:
        if st.button("📜 Attendance History", use_container_width=True):
            st.session_state.show_history = True
            st.rerun()
    with b4:
        if st.button("🔄 Reload Students List", use_container_width=True):
            load_known_faces.clear()
            st.rerun()
            
    st.markdown("---")

    # ----------------------------------------------------------------------
    # Existing Camera Feed and Session Report Sections
    # ----------------------------------------------------------------------
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Classroom Camera Feed")
        frame_placeholder = st.empty()
        status_text = st.empty()

        if st.session_state.is_running:
            # (Your existing camera feed logic remains unchanged here)
            status_text.info("Live camera feed is active.")
            while st.session_state.is_running and st.session_state.cap and st.session_state.cap.isOpened():
                ret, frame = st.session_state.cap.read()
                if not ret:
                    status_text.error("Failed to capture video feed.")
                    st.session_state.is_running = False
                    break
                st.session_state.last_frame = frame.copy()
                small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                if st.session_state.known_students:
                    current_presence = {sid: False for sid in st.session_state.tracker.students}
                    for encoding, location in zip(face_encodings, face_locations):
                        best_match_id, best_match_name = "Unknown", "Unknown"
                        best_distance = 1.0
                        for sid, sdata in st.session_state.known_students.items():
                            distances = face_recognition.face_distance(sdata["encodings"], encoding)
                            min_dist = np.min(distances) if distances.size > 0 else 1.0
                            if min_dist < tolerance and min_dist < best_distance:
                                best_distance, best_match_id, best_match_name = min_dist, sid, sdata["name"]
                        color = (0, 255, 0) if best_match_id != "Unknown" else (0, 0, 255)
                        if best_match_id != "Unknown": current_presence[best_match_id] = True
                        top, right, bottom, left = [int(v / RESIZE_SCALE) for v in location]
                        cv2.rectangle(frame, (left, top), (right, bottom), color, FRAME_THICKNESS)
                        label = f"{best_match_name}" + (f" ({best_match_id})" if best_match_id != "Unknown" else "")
                        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, FONT_THICKNESS)
                    current_time = time.time()
                    for sid in st.session_state.tracker.students:
                        st.session_state.tracker.update_presence(sid, current_presence.get(sid, False), current_time)
                mins, secs = divmod(st.session_state.remaining_time, 60)
                timer_text = f"Session Time: {mins:02d}:{secs:02d}"
                cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                frame_placeholder.image(frame, channels="BGR")
        else:
            if st.session_state.last_frame is not None:
                frame_placeholder.image(st.session_state.last_frame, channels="BGR")
                status_text.info("Session ended. Displaying last captured frame.")
            else:
                frame_placeholder.image(Image.new("RGB", (640, 480), (50, 50, 50)), caption="Camera is off")
                status_text.info("Start a session to begin tracking.")
    
    with col2:
        st.subheader("Session Report")
        if st.session_state.is_running:
            display_live_dashboard(st.session_state.session_start)
        elif st.session_state.csv_data:
            st.success("Session completed!")
            df = pd.DataFrame(st.session_state.csv_data)
            st.dataframe(df)
            csv_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_classroomReport.csv"
            csv_data = df.to_csv(index=False).encode("utf-8")
            save_path = os.path.join(REPORTS_DIR, csv_filename)
            with open(save_path, "wb") as f:
                f.write(csv_data)
            st.download_button("Download Session Report", csv_data, csv_filename, "text/csv", use_container_width=True)
            if st.button("Clear Session Data", use_container_width=True):
                st.session_state.csv_data = None
                st.session_state.last_frame = None
                st.rerun()
        else:
            st.info("The session report will appear here after the session ends.")
            if not st.session_state.known_students:
                st.warning("No students registered. Please register a student to start a session.")
if __name__ == "__main__":
    main()