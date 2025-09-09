import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import csv
import time
from datetime import datetime , timedelta
from PIL import Image
import pandas as pd
import threading

#Configuration
KNOWN_FACES_DIR = "students_faces"
TOLERANCE = 0.5
FRAME_THICKNESS = 2
FONT_THICKNESS = 1
RESIZE_SCALE = 0.25
SESSION_DURATION = 45 * 60 #45 minutes in  seconds(45 mins class)

#Uniform Detection
WHITE_LOWER = np.array([0, 0, 200], dtype=np.uint8)
WHITE_UPPER = np.array([180,25,255] , dtype = np.uint8)
UNIFORM_THRESHOLD = 0.5

#Create Student_Faces Dir , If it does not exist
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

#Initialize Session State
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
    if "session_timer"  not in st.session_state:
        st.session_state.session_timer = None
    if "remaining_time" not in st.session_state:
        st.session_state.remaining_time = SESSION_DURATION

#Load Known Students
def load_known_faces():
    st.session_state.known_students = {}

    for folder in os.listdir(KNOWN_FACES_DIR):
        folder_path = os.path.join(KNOWN_FACES_DIR , folder)
        if not os.path.isdir(folder_path):
            continue

        parts = folder.split("_")
        if len(parts ) >= 2:
            student_id = parts[0]
            name = " ".join(parts [1:])
        else:
            student_id = parts[0]
            name = folder
        
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder_path , img_file)
                try:
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        if student_id not in st.session_state.known_students:
                            st.session_state.known_students[student_id] = {
                                "name" : name,
                                "encodings": []
                            }
                        st.session_state.known_students[student_id][encodings].append(encodings[0])
                except Exception as e:
                   st.error(f"Error Processing {img_file} in {folder} : {e}")
    if not st.session_state.known_students:
        st.warning("No known faces loaded . Add student images to the folders.")
    else:
        st.success(f"Loaded {len(st.session_state.known_students)} students.")

#Update uniform detection function
def check_uniform(face_location, frame):
    try:
        top , right , bottom , left =  face_location
        face_height =  bottom - top

        #Estimate shirt region
        shirt_top = bottom
        shirt_bottom = min(bottom + int(face_height * 1.5), frame.shape[0])
        shirt_left = max(0 , left - int(face_height * 0.3))
        shirt_right = min(frame.shape[1] , right + int(face_height * 0.3))

        shirt_roi = frame[shirt_top : shirt_bottom , shirt_left : shirt_right]
        if shirt_roi == 0:
            return False
        
        #Blur to smooth out small color variations(like buttons or collar)
        blurred = cv2.GaussianBlur(shirt_roi,(7,7) ,0 ) 

        #Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        #Loosen the white threshold range(more tolerant)
        loose_white_lower = np.array([0,0,160], dtype = np.uint8)
        loose_white_upper = np.array([180,60,255], dtype = np.uint8)

        #Create mask and calculate ratio
        mask = cv2.inRange(hsv , loose_white_lower , loose_white_upper)
        white_ratio = np.count_nonzero(mask) / (mask.size + 1e-6)
        

        return white_ratio >= UNIFORM_THRESHOLD
    except Exception as e:
        return False
    
#Track student presence with uniform detection
class StudentTracker:
    def __init__(self):
        self.students ={}
        for student_id , student_data in st.session_state.known_students.items():
            self.students[student_id] ={
                "name":False ,
                "in_frame": None ,
                "start_time": 0.0 ,
                "total_time": None ,
                "first_seen": None ,
                "last_seen": None ,
                "uniform_ok": None ,
                "time_in": None,
                "time_out": None,
            }
    def update_presence(self,student_id, in_frame , current_time):
        student = self.students.get(student_id)
        if not student:
            return
        
        if in_frame and not student["in_frame"]:
            #student just entered frame
            student["in_frame"] = True
            student["start_time"] = current_time

            if student["first_seen"] is None:
                student["first_seen"] = current_time
                student["time_in"] = datetime.frontimestamp(current_time).strftime(
                "%H : %M : %S"
            )

            student["last_seen"] = current_time
        
        elif not in_frame and student["in_frame"]:
            # Student just left frame
            student["in_frame"] = False
            if student["start_time"]:
              duration = current_time - student["start_time"]
              student["total_time"] += duration
              student["start_time"] = None

            student["time_out"] = datetime.fromtimestamp(current_time).strftime(
                 "%H:%M:%S"
            )

    def final_update(self, current_time):
        for student_id, data in self.students.items():
            if data["in_frame"] and data["start_time"]:
                duration = current_time - data["start_time"]
                data["total_time"] += duration
                data["in_frame"] = False
                data["start_time"] = None
                data["time_out"] = datetime.fromtimestamp(current_time).strftime(
                    "%H:%M:%S"
                    )
            
            if data["last_seen"]:
                data["last_seen"] = datetime.fromtimestamp(current_time).strftime(
                         "%H:%M:%S"
                )
    def get_csv_data(self):
        session_start = st.session_state.session_start
        session_end = time.time()

        session_start_dt = datetime.fromtimestamp(session_start)
        session_end_dt = datetime.fromtimestamp(session_end)
        session_date = session_start_dt.strftime("%Y-%m-%d")
        session_start_str = session_start_dt.strftime("%H:%M:%S")
        session_end_str = session_end_dt.strftime("%H:%M:%S")

        session_duration_secs = session_end - session_start
        data = []

        for student_id, info in self.students.items():
            # Check if 'total_time' exists and is a number
            if isinstance(info.get("total_time"), (int, float)):
                total_time_secs = round(info["total_time"], 2)
                total_time_mins = round(total_time_secs / 60, 2)
            else:
                # Assign a default value if 'total_time' is missing or not a number
                total_time_secs = 0.0
                total_time_mins = 0.0

            # Handle students who never entered frame
            if total_time_secs == 0:
                performance = "Absent"
                status = "Absent"
            
            else:
                # Calculate performance ratio
                time_ratio = (
                    total_time_secs / session_duration_secs
                    if session_duration_secs > 0
                    else 0
                )

            # Assign performance bucket
            if time_ratio >= 0.75:
                performance = "Excellent"
            elif time_ratio >= 0.50:
                performance = "Very Good"
            elif time_ratio >= 0.25:
                performance = "Good"
            else:
                performance = "Poor"

            # Determine presence status
            status = "Present" if time_ratio >= 0.50 else "Absent"

            # Uniform Check
            if info["uniform_ok"] is True:
                uniform_status = "OK"
            elif info["uniform_ok"] is False:
                uniform_status = "Violation"
            else:
                uniform_status = "Not Checked"
            
            data.append(
                {
                    "Student ID": student_id,
                    "Name": info["Name"],
                    "Session Start Time": session_start_str,
                    "Session End Time": session_end_str,
                    "Student Class Entering Time": info.get("time_in", "N/A"),
                    "Student Last Seen Time": info.get("last_seen", "N/A"),
                    "Student Check Out Time": info.get("time_out", "N/A"),
                    "Total Time (seconds)": total_time_secs,
                    "Total Time (minutes)": total_time_mins,
                    "Performance": performance,
                    "Status": status,
                    "Uniform Check": uniform_status,
                    "Session Date": session_date,
                }
            )

        return data

# Session timer thread
def session_timer():
    try:
        while(
            st.session_state.get("is_running", False)
            and st.session_state.get("remaining_time", 0) > 0
        ):
            time.sleep(1)
            st.session_state.remaining_time -= 1

        if st.session_state.remaining_time <= 0:
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

# Main Application
def main():
    frame = None
    # Initialize the camera capture object outside the if block
    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)
    st.set_page_config(page_title="Classroom Tracker", page_icon="👨‍🎓", layout="wide")

    # Inside your main loop
    
    ret, frame = st.session_state.cap.read()


    # Load Your Logo
    st.markdown("<h1>👨‍🎓 Student Tracking System </h1>", unsafe_allow_html=True)
    st.markdown(
        """
        **Student classroom monitoring system** that:
        - Automatically marks student attendance
        - Checks uniform compliance (White Shirts)
        - Tracks time in classroom
        - Records entry/exit times
        - Generate session reports
        """
    )

    init_session_state()
    load_known_faces() # Load faces on startup

    # Sidebar for controls
    with st.sidebar:

        st.markdown(
            "<img src='your_logo.jpg' style='width:150px; display:block; margin:auto;'>",
            unsafe_allow_html=True
        )
        st.header("Session Controls")

        # Session timer display
        if st.session_state.is_running:
            mins, secs = divmod(st.session_state.remaining_time, 60)
            timer_text = f"Time Remaining: {mins:02d}:{secs:02d}"
            st.success(timer_text)
        else:
            st.info(f"Session Duration: {SESSION_DURATION//60} minutes")

        st.divider()

        # Student Management
        st.subheader("Student Management")
        if st.button("Reload Student Faces"):
            load_known_faces()
            st.success("Student faces reloaded!")

        # Upload multiple images for one student at once
        uploaded_files = st.file_uploader(
            "Upload multiple images for a student (e.g., 001_Edward_Stien_1.jpg)",
            accept_multiple_files=True,
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                name_id = os.path.splitext(filename)[0] # Remove Extension
                parts = name_id.split("_")

                if len(parts) < 3:
                    st.error(
                        f"Invalid filename: {filename}. Format must be like 001_Edward_Stien_1.jpg"
                    )
                    continue

                student_id = parts[0]
                full_name = "_".join(parts[1:-1]) # All parts except ID and index
                folder_name = f"{student_id}_{full_name.replace(' ', '_')}" # Folder = 001_EdwardStein (optional)

                student_dir = os.path.join(KNOWN_FACES_DIR, folder_name)
                os.makedirs(student_dir, exist_ok=True)

                # Count existing images
                existing_images = [
                    f
                    for f in os.listdir(student_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                new_index = len(existing_images) + 1
                ext = os.path.splitext(filename)[1]
                new_filename = f"{new_index}{ext}"
                save_path = os.path.join(student_dir, new_filename)

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(f"Saved: {folder_name}/{new_filename}")

        load_known_faces()

    st.divider()

    # Start / Stop Tracking
    st.subheader("Session Control")

    if not st.session_state.is_running:
        if st.button(
            "Start Classroom Session",
            use_container_width=True,
            disabled=not st.session_state.known_students,
        ):
            # Initialize tracker
            st.session_state.tracker = StudentTracker()
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.is_running = True
            st.session_state.session_start = time.time()
            st.session_state.remaining_time = SESSION_DURATION

            # Start Session Timer Thread
            st.session_state.session_timer = threading.Thread(target=session_timer)
            st.session_state.session_timer.start()
            st.rerun()

        elif not st.session_state.known_students:
            st.warning("Add student images before starting session")
    else:
        if st.button("End Session Now", use_container_width=True):
            st.session_state.is_running = False

            if st.session_state.tracker:
                current_time = time.time()
                st.session_state.tracker.final_update(current_time)
                st.session_state.csv_data = st.session_state.tracker.get_csv_data()

            if st.session_state.cap and st.session_state.cap.isOpened():
                st.session_state.cap.release()
                
            st.rerun()

    st.divider()
    
    # Display Known Students
    st.subheader("Registered Students")

    if st.session_state.known_students:
        for student_id, student_data in st.session_state.known_students.items():
            st.markdown(f"**{student_data['name']}** (ID: {student_id})")
    else:
        st.info("No Students registered yet")

    # Main Content Area
    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("Classroom Camera Feed")
        frame_placeholder = st.empty()
        
        status_text = st.empty()

        if st.session_state.is_running:
            prev_frame_time = 0

        while st.session_state.is_running and st.session_state.cap.isOpened():
            # Capture Frame
           
            if not ret:
                status_text.error("Failed to capture video feed")
                st.session_state.is_running = False
                break

            # Store last frame for display when not running
            st.session_state.last_frame = frame.copy()

            # Resize for faster processing
            small_frame = cv2.resize(
                frame, (0,0), fx=RESIZE_SCALE, fy=RESIZE_SCALE
            )
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find faces in Frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            # Only proceed if we have known students
            if st.session_state.known_students:
                # Reset presence for this frame
                current_presence = {
                    student_id: False
                    for student_id in st.session_state.tracker.students
                }
            
            # process each face in frame
            for current_encoding, face_location in zip(
                face_encodings, face_locations
            ):
                try:
                    best_match_id = None
                    best_match_name = None
                    best_distance = 1.0

                    for (
                        student_id,
                        student_data,
                    ) in st.session_state.known_students.items():
                        name = student_data["name"]
                        encodings = student_data["encodings"]
                    
                        distances = face_recognition.face_distance(
                            encodings, face_encodings
                        )
                        if len(distances) == 0:
                            continue

                        min_dist = np.min(distances)

                        if (
                            min_dist < TOLERANCE
                            and min_dist < best_distance
                        ):
                            best_distance = min_dist
                            best_match_id = student_id
                            best_match_name = name

                    if best_match_id:
                        current_presence[best_match_id] = True

                        # Scale face location to original frame size
                        top, right, bottom, left = face_location
                        top = int(top / RESIZE_SCALE)
                        right = int(right / RESIZE_SCALE)
                        bottom = int(bottom / RESIZE_SCALE)
                        left = int(left / RESIZE_SCALE)
                        scaled_location = (top, right, bottom, left)

                        # Check Uniform
                        uniform_ok = check_uniform(scaled_location, frame)

                        # Update uniform status in tracker
                        current_status = st.session_state.tracker.students[
                            best_match_id
                        ]["uniform_ok"]

                        if current_status is None or current_status:
                            st.session_state.tracker.students[
                                best_match_id
                            ]["uniform_ok"] = uniform_ok

                        # Draw rectangle and label
                        color = (0, 255, 0) if uniform_ok else (0, 0, 255)
                        cv2.rectangle(
                            frame,
                            (left, top),
                            (right, bottom), # This part was inferred as it was cut off in the image
                            color,
                            FRAME_THICKNESS,
                        )

                        label = f"{best_match_name} (({best_match_id}))"
                        cv2.putText(
                            frame,
                            label,
                            (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            FONT_THICKNESS,
                        )

                        # Add uniform status
                        status = (
                            "Uniform: OK"
                            if uniform_ok
                            else "Uniform: VIOLATION"
                        )
                        cv2.putText(
                            frame,
                            status,
                            (left, bottom + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, 
                            color,
                            FONT_THICKNESS,
                            0.5,
                            color,
                            FONT_THICKNESS,
                        )

                except Exception as e:
                    status_text.error(f"Face processing error: {str(e)}")
            
            # Update presence in tracker
            current_time = time.time()
            for student_id in st.session_state.tracker.students:
                st.session_state.tracker.update_presence(
                    student_id, current_presence[student_id], current_time
                )
            
        # Display session info on frame
        mins, secs = divmod(st.session_state.remaining_time, 60)
        timer_text = f"Session Time: {mins:02d}:{secs:02d}"
        cv2.putText(
            frame,
            timer_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # Display frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")


        # When loop exits, show last frame
        if st.session_state.last_frame is not None:
            frame_placeholder.image(st.session_state.last_frame, channels="BGR")
        else:
            # Show placeholder when not tracking
            if st.session_state.last_frame is not None:
                frame_placeholder.image(st.session_state.last_frame, channels="BGR")
            else:
                # Display a camera placeholder
                camera_placeholder = Image.new("RGB", (640, 480), (50, 50, 50))
                frame_placeholder.image(camera_placeholder)
            status_text.info("Camera feed will appear when session starts")
    
    with col2:
        st.subheader("Session Report")

        if st.session_state.is_running:
            st.info("Tracking in progress...")

            # Display active tracking status
            if hasattr(st.session_state, "tracker") and st.session_state.tracker:
                for student_id, data in st.session_state.tracker.students.items():
                    time_min = data["total_time"] / 60
                    if data["in_frame"] and data["start_time"]:
                        time_min += (time.time() - data["start_time"]) / 60

                    status = "Present" if data["in_frame"] else "Not in Frame"
                    
                    # Uniform status
                    uniform_status = "Not Checked"
                    if data["uniform_ok"] is True:
                        uniform_status = "Uniform OK"
                    elif data["uniform_ok"] is False:
                        uniform_status = "Uniform Violation"
                    
                    with st.expander(f"**{data['name']}** ({student_id})"):
                        st.caption(f"{status} | {time_min:.2f} minutes")
                        st.caption(f"{uniform_status}")
                        if data['time_in']:
                            st.caption(f"Time In: {data['time_in']}")
                        if data['time_out']:
                            st.caption(f"Last Exit: {data['time_out']}")

        elif st.session_state.csv_data:
            st.success("Session completed!")
            
            # Show CSV data
            df = pd.DataFrame(st.session_state.csv_data)
            st.dataframe(df)

            # Create downloadable CSV
            csv_filename = f"{datetime.now().strftime('%Y%m%d')}_classroomReport.csv"
            csv_data = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Session Report",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
                use_container_width=True,
            )

            if st.button("Clear Session Data", use_container_width=True):
                st.session_state.csv_data = None
                st.session_state.last_frame = None
                st.rerun()

        else:
            st.info("Start a classroom session to begin tracking")

            if st.session_state.known_students:
                st.subheader("Ready to Track")
                st.markdown("The Following Students are Registered:")

                for student_id, student in st.session_state.known_students.items():
                    st.markdown(f"- **{student['name']}** (ID: {student_id})")

            else:
                st.warning('No student faces availbale. Add images using the sidebar.')

if __name__ == "__main__":
    main()

                





    
              


    




            

