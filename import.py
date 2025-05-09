import serial
import time
import hashlib
import json

# Define the serial port where Arduino is connected
PORT = 'COM16'  # Adjust to your system's serial port
BAUD_RATE = 57600

# Path to store the JSON file for fingerprints
JSON_FILE = 'fingerprints.json'

# Load existing fingerprints from the JSON file (if any)
def load_fingerprints():
    try:
        with open(JSON_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Save the fingerprints to the JSON file
def save_fingerprints(fingerprints):
    with open(JSON_FILE, 'w') as f:
        json.dump(fingerprints, f, indent=4)

# Generate a hash of the fingerprint data (for comparison)
def hash_fingerprint(fingerprint_data):
    return hashlib.sha256(fingerprint_data).hexdigest()

# Initialize serial connection to Arduino
def initialize_arduino():
    return serial.Serial(PORT, BAUD_RATE)

# Function to capture fingerprint data from Arduino
def get_fingerprint_from_arduino(arduino):
    arduino.write(b'get')  # Request fingerprint from Arduino
    time.sleep(2)
    
    # Read the response from Arduino (this should be the fingerprint template)
    if arduino.in_waiting > 0:
        fingerprint_data = arduino.read(arduino.in_waiting)
        return fingerprint_data
    else:
        return None

# Function to register a new fingerprint
def register_fingerprint(arduino, fingerprints):
    # Ask for user details
    name = input("Enter Name: ")
    roll_no = input("Enter Roll Number: ")

    print("Place your finger on the sensor...")
    fingerprint_data = get_fingerprint_from_arduino(arduino)
    
    if fingerprint_data:
        print("Fingerprint captured!")
        fingerprint_hash = hash_fingerprint(fingerprint_data)
        
        # Check if the fingerprint hash already exists
        if fingerprint_hash not in fingerprints:
            fingerprints[fingerprint_hash] = {'name': name, 'roll_no': roll_no}
            save_fingerprints(fingerprints)
            print("Fingerprint registered successfully!")
        else:
            print("This fingerprint is already registered.")
    else:
        print("Failed to capture fingerprint.")

# Function to recognize an existing fingerprint
def recognize_fingerprint(arduino, fingerprints):
    print("Place your finger on the sensor...")
    fingerprint_data = get_fingerprint_from_arduino(arduino)
    
    if fingerprint_data:
        print("Fingerprint captured!")
        fingerprint_hash = hash_fingerprint(fingerprint_data)

        # Check if the fingerprint hash exists in the database
        if fingerprint_hash in fingerprints:
            user_info = fingerprints[fingerprint_hash]
            print(f"Hello, {user_info['name']}! Roll Number: {user_info['roll_no']}")
        else:
            print("No match found.")
    else:
        print("Failed to capture fingerprint.")

def main():
    # Initialize Arduino connection
    arduino = initialize_arduino()

    # Load existing fingerprints
    fingerprints = load_fingerprints()

    while True:
        action = input("Enter 'register' to register a new fingerprint or 'recognize' to recognize a fingerprint: ").strip().lower()
        
        if action == 'register':
            register_fingerprint(arduino, fingerprints)
        elif action == 'recognize':
            recognize_fingerprint(arduino, fingerprints)
        else:
            print("Invalid option. Please enter 'register' or 'recognize'.")
        
        continue_prompt = input("Do you want to continue? (y/n): ").strip().lower()
        if continue_prompt != 'y':
            break

    arduino.close()

if __name__ == "__main__":
    main()
