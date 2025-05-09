import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import csv
from flask import Flask, render_template, Response, jsonify, request
import pickle
from flask_socketio import SocketIO
import json

app = Flask(__name__, template_folder='template')
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)


TRAINING_PATH = r"D:\Rahul SD card data\zzprojects\1Dl\images"
ATTENDANCE_FILE = r"D:\Rahul SD card data\zzprojects\1Dl\Attendance.csv"
ENCODINGS_FILE = "encodings.pkl"
ABSENTEES_FILE = r"D:\Rahul SD card data\zzprojects\1Dl\Absentees.csv"
THRESHOLD = 0.40  
RESIZE_FACTOR = 0.5  

PERIOD_TIMINGS = [
    ("1", "00:10", "10:00"),
    ("2", "10:00", "10:50"),
    ("3", "11:10", "12:00"),
    ("4", "12:00", "22:50"),
    ("5", "22:50", "23:40"),
    ("6", "23:40", "23:50"),
    ("7", "23:50", "23:53"),
    ("8", "23:53", "07:55"),
]

with open('timetable.json', 'r') as file:
    TIMETABLE = json.load(file)

attendance_time_tracker = {}  
present_students = set()
all_students = [] 


def initialize_files():
    if not os.path.isfile(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Register Number', 'Time', 'Day', 'Date', 'Period', 'Subject', 'Staff', 'Code'])
    if not os.path.isfile(ABSENTEES_FILE):
        with open(ABSENTEES_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Register Number', 'Day', 'Date', 'Period', 'Subject', 'Staff', 'Code'])

folder_path = r"C:\1Dl"
os.makedirs(folder_path, exist_ok=True)


def save_encodings(encodings, names):
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((encodings, names), f)


def load_encodings():
    if os.path.isfile(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return [], []


def extract_student_info(folder_name):
    if '-' in folder_name:
        parts = folder_name.split('-')
        if len(parts) >= 2:
            name = parts[0].strip()
            reg_number = parts[1].strip()
            return name, reg_number
    return folder_name, ""

def load_training_images(path):
    images = []
    names = []
    total_students = []
    
    for folder_name in os.listdir(path):
        if folder_name.startswith('.'):
            continue
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path):
            if '-' in folder_name:  
                name, reg_number = extract_student_info(folder_name)
                student_info = {"name": name, "register_number": reg_number, "full_id": folder_name}
                total_students.append(student_info)
                
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
    
    global all_students
    all_students = total_students
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


def get_current_timetable_entry():
    now = datetime.now()
    day = now.strftime('%A')
    current_time = now.strftime('%H:%M')
    
    for period, start, end in PERIOD_TIMINGS:
        if start <= current_time <= end:
            day_timetable = TIMETABLE.get(day, [])
            for entry in day_timetable:
                if entry["period"] == int(period):
                    return period, entry
    return None, None


def mark_attendance(student_id, period, subject, staff, code):
    name, register_number = extract_student_info(student_id)
    now = datetime.now()
    attendance_time_tracker[student_id] = now
    
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, register_number, now.strftime('%I:%M:%S %p'),
                         now.strftime('%A'), now.strftime('%d-%m-%Y'), period, subject, staff, code])
    
    socketio.emit('update_attendance', {
        'name': name,
        'register_number': register_number,
        'time': now.strftime('%I:%M:%S %p'),
        'day': now.strftime('%A'),
        'date': now.strftime('%d-%m-%Y'),
        'period': period,
        'subject': subject,
        'staff': staff,
        'code': code
    })


@app.route('/video_feed')
def video_feed():
    def generate_frames():
        known_encodings, known_names = load_encodings()
        if not known_encodings:
            images, names, _ = load_training_images(TRAINING_PATH)
            known_encodings = find_encodings(images)
            save_encodings(known_encodings, names)

        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Error: Unable to access the camera.")
            return

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            current_period, timetable_entry = get_current_timetable_entry()
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(distances) > 0:
                    min_distance = min(distances)
                    if min_distance < THRESHOLD:
                        match_index = np.argmin(distances)
                        student_id = known_names[match_index]
                        name, reg_number = extract_student_info(student_id)
                        
                        if timetable_entry and student_id not in present_students:
                            present_students.add(student_id)
                            mark_attendance(student_id, str(timetable_entry["period"]), timetable_entry["subject"],
                                            timetable_entry["staff"], timetable_entry["code"])

                        
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        
                        
                        y_position = top - 10
                        if y_position < 10:
                            y_position = bottom + 20
                            

                            
                        cv2.putText(frame, f"{name}", (left, y_position-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"{reg_number}", (left, y_position + 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        video_capture.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('in.html')

@app.route('/attendance')
def get_attendance():
    try:
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.DictReader(f)
            attendance_data = list(reader)
            return jsonify(attendance_data)
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        return jsonify({'error': f'Failed to load attendance data: {str(e)}'})

@app.route('/present_students')
def get_present_students():
    result = []
    for student_id in present_students:
        name, reg_number = extract_student_info(student_id)
        result.append({
            'name': name,
            'register_number': reg_number
        })
    
   
    result.sort(key=lambda x: x['register_number'])
    return jsonify(result)

@app.route('/reset_attendance', methods=['POST'])
def reset_attendance():
    global present_students
    present_students = set()
    return jsonify({'status': 'success'})
@app.route('/current_period')
def get_current_period_info():
    period, timetable_entry = get_current_timetable_entry()
    if period and timetable_entry:
        return jsonify({
            'period': period,
            'subject': timetable_entry['subject'],
            'staff': timetable_entry['staff'],
            'code': timetable_entry['code']
        })
    return jsonify({
        'period': '-',
        'subject': '-',
        'staff': '-',
        'code': '-'
    })
@app.route('/attendance_stats')
def get_attendance_stats():
    
    total_students = len(all_students)
    
    
    present_count = len(present_students)
    
    
    absent_count = total_students - present_count
    
    
    attendance_rate = (present_count / total_students * 100) if total_students > 0 else 0
    
    
    historical_rates = []
    try:
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.DictReader(f)
            
            attendance_by_date = {}
            for row in reader:
                date = row['Date']
                if date not in attendance_by_date:
                    attendance_by_date[date] = []
                attendance_by_date[date].append(row)
            
            
            for date, records in attendance_by_date.items():
                daily_rate = len(set([r['Register Number'] for r in records])) / total_students * 100
                historical_rates.append(daily_rate)
    except Exception as e:
        print(f"Error reading historical data: {e}")
    
    
    avg_attendance = sum(historical_rates) / len(historical_rates) if historical_rates else attendance_rate
    highest_attendance = max(historical_rates) if historical_rates else attendance_rate
    lowest_attendance = min(historical_rates) if historical_rates else attendance_rate
    
    
    trend = "Stable"
    if len(historical_rates) >= 2:
        if historical_rates[-1] > historical_rates[0]:
            trend = "Improving"
        elif historical_rates[-1] < historical_rates[0]:
            trend = "Declining"
    
    return jsonify({
        'total_students': total_students,
        'present_count': present_count,
        'absent_count': absent_count,
        'attendance_rate': round(attendance_rate, 1),
        'avg_attendance': round(avg_attendance, 1),
        'highest_attendance': round(highest_attendance, 1),
        'lowest_attendance': round(lowest_attendance, 1),
        'trend': trend
    })
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    initialize_files()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)