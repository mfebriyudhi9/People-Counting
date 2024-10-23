import numpy as np
import json
import cv2
import os
from mtcnn import MTCNN
import time

def create_directory(directory: str) -> None:
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_face_id(directory: str) -> int:
    """Get the first available identifier by scanning the filenames."""
    user_ids = []
    for filename in os.listdir(directory):
        if '-' in filename:
            try:
                number = int(os.path.split(filename)[-1].split("-")[1])
                user_ids.append(number)
            except (IndexError, ValueError):
                print(f"Skipping invalid file: {filename}")
                continue  # Skip files that don't match the expected pattern

    user_ids = sorted(set(user_ids))
    max_user_id = 1 if not user_ids else max(user_ids) + 1
    return max_user_id

def save_name(face_id: int, face_name: str, filename: str) -> None:
    """Save name on JSON of names."""
    names_json = {}
    if os.path.exists(filename):
        with open(filename, 'r') as fs:
            names_json = json.load(fs)
    names_json[face_id] = face_name
    with open(filename, 'w') as fs:
        json.dump(names_json, fs, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    directory = 'images_muka'
    names_json_filename = 'names.json'

    # Create 'images' directory if it doesn't exist
    create_directory(directory)

    # Initialize MTCNN detector
    detector = MTCNN()

    # Open a connection to the default camera (camera index 0)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Camera not found or cannot be opened.")
        exit()

    cam.set(3, 640)  # Set width
    cam.set(4, 480)  # Set height

    # Ask for user name
    face_name = input('\nEnter user name and press <return> -->  ')
    face_id = get_face_id(directory)
    save_name(face_id, face_name, names_json_filename)
    print('\n[INFO] Initializing face capture. Look at the camera and wait...')

    # Initialize count and timing
    count = 0
    start_time = time.time()
    capture_duration = 60  # 1 minute duration

    while count < 50 and (time.time() - start_time) < capture_duration:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Failed to capture image")
            break

        # Detect faces using MTCNN
        results = detector.detect_faces(img)

        if results:
            for result in results:
                x, y, w, h = result['box']
                face = img[y:y+h, x:x+w]
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Increment the count for naming the saved images
                count += 1

                # Save the captured image into the 'images' directory
                cv2.imwrite(f'./images_muka/Users-{face_id}-{count}.jpg', gray_face)
                cv2.imshow('image', img)

                # Break if 50 images are captured
                if count >= 50:
                    break

        # Check if time is up or exit key is pressed
        if cv2.waitKey(100) & 0xff == 27:
            break

    print('\n[INFO] Success! Exiting Program.')

    # Release the camera and close all OpenCV windows
    cam.release()
    cv2.destroyAllWindows()