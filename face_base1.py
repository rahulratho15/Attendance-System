import cv2
import numpy as np
import face_recognition
import os
import time
from scipy.spatial import distance as dist

TRAINING_PATH = r"D:\Rahul SD card data\zzprojects\1Dl\images"
THRESHOLD = 0.40  
RESIZE_FACTOR = 0.5  

# Liveness detection parameters
EYE_AR_THRESH = 0.2        # Eye aspect ratio threshold for blink detection
EYE_AR_CONSEC_FRAMES = 3   # Number of consecutive frames the eye must be below threshold to be considered a blink
BLINK_TIMEOUT = 5.0        # Time in seconds after which a blink must be detected
EYE_MOVEMENT_THRESH = 2.0  # Threshold for eye movement detection (in pixels)
MOVEMENT_TIMEOUT = 3.0     # Time in seconds after which eye movement must be detected
LIVENESS_VALID_TIME = 30.0 # Time in seconds that liveness verification remains valid after detection

# Global variables for liveness detection
blink_counter = 0
last_blink_time = None
last_eye_positions = []
last_movement_time = None
liveness_verified = False
liveness_verified_time = None
present_students = set()  # To track students who have been recognized

# Helper functions to load student info
def extract_student_info(folder_name):
    if '-' in folder_name:
        parts = folder_name.split('-')
        if len(parts) >= 2:
            name = parts[0].strip()
            reg_number = parts[1].strip()
            return name, reg_number
    return folder_name, ""  # Default return if no '-' in the folder name

def load_training_images(path):
    images = []
    names = []
    
    for folder_name in os.listdir(path):
        if folder_name.startswith('.'):
            continue
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path):
            if '-' in folder_name:  # Only process valid folders with student name and reg number
                name, reg_number = extract_student_info(folder_name)
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img = cv2.imread(file_path)
                        if img is not None:
                            img = cv2.resize(img, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
                            images.append(img)
                            names.append(folder_name)  # Store the folder name as student ID
            else:
                print(f"Skipping folder '{folder_name}' due to invalid naming format.")
    
    return images, names

# Function to find face encodings
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

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Function to get eye positions (center of each eye)
def get_eye_positions(eye_landmarks):
    left_eye = eye_landmarks[36:42]
    right_eye = eye_landmarks[42:48]
    
    # Calculate the center of each eye
    left_eye_center = np.mean(left_eye, axis=0).astype(int)
    right_eye_center = np.mean(right_eye, axis=0).astype(int)
    
    return left_eye_center, right_eye_center

# Function to detect eye movement
def detect_eye_movement(current_eye_positions):
    global last_eye_positions, last_movement_time
    
    # If we don't have previous eye positions, store the current ones and return
    if not last_eye_positions:
        last_eye_positions = current_eye_positions
        last_movement_time = time.time()
        return False
    
    # Calculate the movement distance for each eye
    left_distance = dist.euclidean(current_eye_positions[0], last_eye_positions[0])
    right_distance = dist.euclidean(current_eye_positions[1], last_eye_positions[1])
    
    # Update last eye positions
    last_eye_positions = current_eye_positions
    
    # If either eye moved more than the threshold, consider it as eye movement
    if left_distance > EYE_MOVEMENT_THRESH or right_distance > EYE_MOVEMENT_THRESH:
        last_movement_time = time.time()
        return True
    
    return False

# Function to check liveness
def check_liveness(face_landmarks):
    global blink_counter, last_blink_time, last_movement_time, liveness_verified, liveness_verified_time
    
    current_time = time.time()
    
    # If liveness was already verified recently, return True
    if liveness_verified and liveness_verified_time and (current_time - liveness_verified_time) < LIVENESS_VALID_TIME:
        return True
    
    # Reset liveness verification
    liveness_verified = False
    
    # Extract the shape of facial landmarks
    facial_landmarks = np.array([[p[0], p[1]] for p in face_landmarks])
    
    # Get left and right eye landmarks
    left_eye = facial_landmarks[36:42]
    right_eye = facial_landmarks[42:48]
    
    # Calculate the eye aspect ratio
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    
    # Average the eye aspect ratio together for both eyes
    ear = (left_ear + right_ear) / 2.0
    
    # Get eye positions for movement detection
    eye_positions = get_eye_positions(facial_landmarks)
    
    # Detect eye movement
    eye_moved = detect_eye_movement(eye_positions)
    
    # Initialize blink times if not set
    if last_blink_time is None:
        last_blink_time = current_time
    if last_movement_time is None:
        last_movement_time = current_time
    
    # Check if the eye aspect ratio is below the blink threshold
    if ear < EYE_AR_THRESH:
        blink_counter += 1
    else:
        # If the eyes were closed for a sufficient number of frames, count it as a blink
        if blink_counter >= EYE_AR_CONSEC_FRAMES:
            last_blink_time = current_time
        # Reset the blink counter
        blink_counter = 0
    
    # Check if both blink and eye movement were detected within the time limits
    blink_detected = (current_time - last_blink_time) < BLINK_TIMEOUT
    movement_detected = (current_time - last_movement_time) < MOVEMENT_TIMEOUT
    
    liveness_verified = blink_detected and movement_detected
    
    if liveness_verified:
        liveness_verified_time = current_time
    
    return liveness_verified

# Main function to run the webcam and detect faces
def run_face_recognition():
    global blink_counter, last_blink_time, last_movement_time, liveness_verified, liveness_verified_time
    
    # Load training images and generate encodings from scratch
    images, names = load_training_images(TRAINING_PATH)
    known_encodings = find_encodings(images)
    
    # Start video capture
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
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
        
        liveness_status = "Liveness: Not Verified"
        
        for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
            
            # Scale the face coordinates back to the original image size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # Scale facial landmarks
            scaled_landmarks = []
            for feature in face_landmarks.values():
                scaled_feature = [(pt[0] * 2, pt[1] * 2) for pt in feature]
                scaled_landmarks.extend(scaled_feature)
            
            # Get eye landmarks for liveness detection
            flat_landmarks = [item for sublist in face_landmarks.values() for item in sublist]
            scaled_flat_landmarks = [(pt[0] * 2, pt[1] * 2) for pt in flat_landmarks]
            
            # Check liveness using eye movement and blink detection
            is_live = check_liveness(scaled_flat_landmarks)
            
            # Update liveness status text
            if is_live:
                liveness_status = "Liveness: Verified"
                
                # Only proceed with face recognition if liveness is verified
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(distances) > 0:
                    min_distance = min(distances)
                    if min_distance < THRESHOLD:
                        match_index = np.argmin(distances)
                        student_id = names[match_index]
                        name, reg_number = extract_student_info(student_id)

                        if student_id not in present_students:
                            present_students.add(student_id)  # Add student to present list

                        # Draw a green box around the face (verified and recognized)
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        
                        # Display name & reg number
                        y_position = top - 10
                        if y_position < 10:
                            y_position = bottom + 20
                        cv2.putText(frame, f"{name}", (left, y_position - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"{reg_number}", (left, y_position + 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        # Face is live but not recognized
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 2)
                        y_position = top - 10
                        if y_position < 10:
                            y_position = bottom + 20
                        cv2.putText(frame, "Unknown", (left, y_position),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                liveness_status = "Liveness: Not Verified (Possible Spoof)"
                # Draw a red box for non-live faces (potential spoof)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                y_position = top - 10
                if y_position < 10:
                    y_position = bottom + 20
                cv2.putText(frame, "Potential Spoof", (left, y_position),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw eye landmarks for visualization
            left_eye_points = [(scaled_flat_landmarks[i][0], scaled_flat_landmarks[i][1]) for i in range(36, 42)]
            right_eye_points = [(scaled_flat_landmarks[i][0], scaled_flat_landmarks[i][1]) for i in range(42, 48)]
            
            for point in left_eye_points + right_eye_points:
                cv2.circle(frame, point, 2, (255, 0, 0), -1)
            
            # Draw eye contours
            cv2.polylines(frame, [np.array(left_eye_points)], True, (0, 255, 255), 1)
            cv2.polylines(frame, [np.array(right_eye_points)], True, (0, 255, 255), 1)
        
        # Display liveness status
        cv2.putText(frame, liveness_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display timing information
        current_time = time.time()
        
        if last_blink_time:
            time_since_blink = current_time - last_blink_time
            blink_text = f"Time since last blink: {time_since_blink:.1f}s"
            cv2.putText(frame, blink_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if last_movement_time:
            time_since_movement = current_time - last_movement_time
            movement_text = f"Time since last eye movement: {time_since_movement:.1f}s"
            cv2.putText(frame, movement_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Display the resulting frame
        cv2.imshow("Face Recognition with Liveness Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close any open windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_face_recognition()