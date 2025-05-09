import cv2
import numpy as np
import face_recognition
import os

TRAINING_PATH = r"D:\Rahul SD card data\zzprojects\1Dl\images"
THRESHOLD = 0.40  
RESIZE_FACTOR = 0.5  

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

# Main function to run the webcam and detect faces
def run_face_recognition():
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

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            # Scale the face coordinates back to the original image size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # Check if face matches any known face
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(distances) > 0:
                min_distance = min(distances)
                if min_distance < THRESHOLD:
                    match_index = np.argmin(distances)
                    student_id = names[match_index]
                    name, reg_number = extract_student_info(student_id)

                    if student_id not in present_students:
                        present_students.add(student_id)  # Add student to present list

                    # Draw a box around the face and display name & reg number
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    y_position = top - 10
                    if y_position < 10:
                        y_position = bottom + 20
                    cv2.putText(frame, f"{name}", (left, y_position - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"{reg_number}", (left, y_position + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Face Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close any open windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_face_recognition()
