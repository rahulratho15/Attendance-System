from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

# Initialize Flask app
app = Flask(__name__)

# Constants
TRAINING_PATH = r"C:\1Dl\images"
ATTENDANCE_FILE = r"C:\1Dl\Attendance.csv"
ABSENTEES_FILE = r"C:\1Dl\Absentees.csv"
THRESHOLD = 0.40  # Distance threshold for face recognition
RESIZE_FACTOR = 0.5  # Resize images to this factor to reduce memory usage

# Timetable for periods
PERIOD_TIMINGS = [
    ("1", "09:10", "10:00"),
    ("2", "10:00", "10:50"),
    ("3", "11:10", "12:00"),
    ("4", "12:00", "22:50"),
    ("5", "22:50", "23:40"),
    ("6", "23:40", "23:50"),
    ("7", "23:50", "23:53"),
    ("8", "23:53", "23:55"),
]

# Load timetable (use your defined TIMETABLE dictionary here)
# Load timetable
TIMETABLE = {
    "Sunday": [
        {"period": 1, "subject": "UI and UX Design", "staff": "Ms. R. Venisha", "code": "CCS370"},
        {"period": 2, "subject": "UI and UX Design", "staff": "Ms. R. Venisha", "code": "CCS370"},
        {"period": 3, "subject": "Disaster Risk Reduction and Management", "staff": "Dr. B. Kiruba", "code": "MX3084"},
        {"period": 4, "subject": "Data and Information Security", "staff": "Dr. S. V. Manisekaran", "code": "CW3551"},
        {"period": 5, "subject": "Distributed Computing", "staff": "Mrs. P. Sujitha", "code": "CS3551"},
        {"period": 6, "subject": "Deep Learning", "staff": "Dr. J. Preethi", "code": "AD3501"},
        {"period": 7, "subject": "Disaster Risk Reduction and Management", "staff": "Dr. B. Kiruba", "code": "MX3084"},
        {"period": 8, "subject": "Engineering Secure Software Systems", "staff": "Ms. R. Jeevitha", "code": "CB3591"},
    ],
    "Monday": [
        {"period": 1, "subject": "UI and UX Design", "staff": "Ms. R. Venisha", "code": "CCS370"},
        {"period": 2, "subject": "UI and UX Design", "staff": "Ms. R. Venisha", "code": "CCS370"},
        {"period": 3, "subject": "Disaster Risk Reduction and Management", "staff": "Dr. B. Kiruba", "code": "MX3084"},
        {"period": 4, "subject": "Data and Information Security", "staff": "Dr. S. V. Manisekaran", "code": "CW3551"},
        {"period": 5, "subject": "Distributed Computing", "staff": "Mrs. P. Sujitha", "code": "CS3551"},
        {"period": 6, "subject": "Deep Learning", "staff": "Dr. J. Preethi", "code": "AD3501"},
        {"period": 7, "subject": "Disaster Risk Reduction and Management", "staff": "Dr. B. Kiruba", "code": "MX3084"},
        {"period": 8, "subject": "Engineering Secure Software Systems", "staff": "Ms. R. Jeevitha", "code": "CB3591"},
    ],
    "Tuesday": [
        {"period": 1, "subject": "Distributed Computing", "staff": "Mrs. P. Sujitha", "code": "CS3551"},
        {"period": 2, "subject": "Deep Learning", "staff": "Dr. J. Preethi", "code": "AD3501"},
        {"period": 3, "subject": "UI and UX Design", "staff": "Ms. R. Venisha", "code": "CCS370"},
        {"period": 4, "subject": "Data and Information Security", "staff": "Dr. S. V. Manisekaran", "code": "CW3551"},
        {"period": 5, "subject": "Disaster Risk Reduction and Management", "staff": "Dr. B. Kiruba", "code": "MX3084"},
        {"period": 6, "subject": "Ethical Hacking", "staff": "Ms. B. S. Revathi", "code": "CCS344"},
        {"period": 7, "subject": "Engineering Secure Software Systems", "staff": "Ms. R. Jeevitha", "code": "CB3591"},
        {"period": 8, "subject": "Engineering Secure Software Systems", "staff": "Ms. R. Jeevitha", "code": "CB3591"},
    ],
    "Wednesday": [
        {"period": 1, "subject": "Big Data Analytics", "staff": "Mr. S. Saravanakumar", "code": "CCS334"},
        {"period": 2, "subject": "Game Development", "staff": "Ms. N. S. Neelaveni", "code": "CCS347"},
        {"period": 3, "subject": "Data and Information Security", "staff": "Dr. S. V. Manisekaran", "code": "CW3551"},
        {"period": 4, "subject": "Deep Learning", "staff": "Dr. J. Preethi", "code": "AD3501"},
        {"period": 5, "subject": "UI and UX Design", "staff": "Ms. R. Venisha", "code": "CCS370"},
        {"period": 6, "subject": "Engineering Secure Software Systems", "staff": "Ms. R. Jeevitha", "code": "CB3591"},
        {"period": 7, "subject": "Big Data Analytics", "staff": "Mr. S. Saravanakumar", "code": "CCS334"},
        {"period": 8, "subject": "Big Data Analytics", "staff": "Mr. S. Saravanakumar", "code": "CCS334"},
    ],
    "Thursday": [
        {"period": 1, "subject": "Naan Mudhalvan", "staff": "N/A", "code": "N/A"},
        {"period": 2, "subject": "Naan Mudhalvan", "staff": "N/A", "code": "N/A"},
        {"period": 3, "subject": "Naan Mudhalvan", "staff": "N/A", "code": "N/A"},
        {"period": 4, "subject": "Naan Mudhalvan", "staff": "N/A", "code": "N/A"},
        {"period": 5, "subject": "Distributed Computing", "staff": "Mrs. P. Sujitha", "code": "CS3551"},
        {"period": 6, "subject": "Ethical Hacking", "staff": "Ms. B. S. Revathi", "code": "CCS344"},
        {"period": 7, "subject": "Ethical Hacking", "staff": "Ms. B. S. Revathi", "code": "CCS344"},
        {"period": 8, "subject": "Game Development", "staff": "Ms. N. S. Neelaveni", "code": "CCS347"},
    ],
    "Friday": [
        {"period": 1, "subject": "Game Development", "staff": "Ms. N. S. Neelaveni", "code": "CCS347"},
        {"period": 2, "subject": "Game Development", "staff": "Ms. N. S. Neelaveni", "code": "CCS347"},
        {"period": 3, "subject": "Ethical Hacking", "staff": "Ms. B. S. Revathi", "code": "CCS344"},
        {"period": 4, "subject": "Big Data Analytics", "staff": "Mr. S. Saravanakumar", "code": "CCS334"},
        {"period": 5, "subject": "Deep Learning Lab", "staff": "Dr. J. Preethi & Dr. N. Fareena", "code": "AD3511"},
        {"period": 6, "subject": "Deep Learning Lab", "staff": "Dr. J. Preethi & Dr. N. Fareena", "code": "AD3511"},
        {"period": 7, "subject": "Deep Learning Lab", "staff": "Dr. J. Preethi & Dr. N. Fareena", "code": "AD3511"},
        {"period": 8, "subject": "Deep Learning Lab", "staff": "Dr. J. Preethi & Dr. N. Fareena", "code": "AD3511"},
    ],
}  

# Global variables for attendance
attendance_time_tracker = {}
present_students = set()
current_period = None
timetable_entry = None


# Load training images and encodings
def load_training_images(path):
    images = []
    names = []
    total_students = []
    for folder_name in os.listdir(path):
        if folder_name.startswith('.'):
            continue
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path):
            total_students.append(folder_name)
            if '-' in folder_name:
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img = cv2.imread(file_path)
                        if img is not None:
                            img = cv2.resize(img, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
                            images.append(img)
                            names.append(folder_name)
            else:
                print(f"Skipping folder '{folder_name}' due to invalid naming format.")
    return images, names, total_students


def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encode_list.append(encode)
        except IndexError:
            print("Warning: No face found in one of the images.")
    return encode_list


# Get current timetable entry
def get_current_timetable_entry():
    now = datetime.now()
    day = now.strftime('%A')  # Get day name
    current_time = now.strftime('%H:%M')
    for period, start, end in PERIOD_TIMINGS:
        if start <= current_time <= end:
            day_timetable = TIMETABLE.get(day, [])
            for entry in day_timetable:
                if entry["period"] == int(period):
                    return period, entry
    return None, None  # Outside of period timings


# Mark attendance
def mark_attendance(name, period, subject, staff, code):
    register_number = name.split('-')[1]
    now = datetime.now()
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            name.split('-')[0], register_number,
            now.strftime('%I:%M:%S %p'),
            now.strftime('%A'), now.strftime('%d-%m-%Y'),
            period, subject, staff, code
        ])
    attendance_time_tracker[name] = now


# Video feed generator
def generate_video_feed(known_encodings, known_names):
    global present_students, current_period, timetable_entry
    video_capture = cv2.VideoCapture(0)

    # Get current period and timetable entry
    current_period, timetable_entry = get_current_timetable_entry()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize and convert the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process faces in the frame
        for face_encoding, face_location in zip(face_encodings, face_locations):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            min_distance = min(distances)
            if min_distance < THRESHOLD:
                match_index = np.argmin(distances)
                name = known_names[match_index]
                if name not in present_students:
                    present_students.add(name)
                    mark_attendance(name, timetable_entry["period"], timetable_entry["subject"],
                                    timetable_entry["staff"], timetable_entry["code"])

                # Draw bounding box and name on the frame
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name.split('-')[0], (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Encode the frame and send it as a response
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()


# Flask routes
@app.route("/")
def index():
    return render_template("index.html")  # Your provided frontend HTML


@app.route("/video_feed")
def video_feed():
    # Load training data
    training_images, training_names, _ = load_training_images(TRAINING_PATH)
    known_encodings = find_encodings(training_images)
    return Response(generate_video_feed(known_encodings, training_names),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/attendance")
def get_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return jsonify(data)
    else:
        return jsonify([])


@app.route("/timetable")
def get_timetable():
    return jsonify(TIMETABLE)


# Initialize the attendance file
def initialize_files():
    if not os.path.isfile(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Register Number', 'Time', 'Day', 'Date', 'Period', 'Subject', 'Staff', 'Code'])


# Main
if __name__ == "__main__":
    initialize_files()
    app.run(host="0.0.0.0", port=5000, debug=True)
