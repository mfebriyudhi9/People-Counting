import cv2
import numpy as np
import json
from mtcnn import MTCNN
import threading

# Load the trained LBPH model and names from JSON
with open('names.json') as f:
    names = json.load(f)

# Initialize MTCNN for face detection
detector = MTCNN()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Variables for counting
people_in_cam0 = 0
people_out_cam1 = 0
detected_faces_cam0 = {}
detected_faces_cam1 = {}

# Flag to stop threads
exit_flag = False

# Function for processing a single camera
def process_camera(camera_id):
    global people_in_cam0, people_out_cam1, detected_faces_cam0, detected_faces_cam1, exit_flag

    cap = cv2.VideoCapture(camera_id)  # Use camera_id to capture from the corresponding camera

    while True:
        # Check if exit flag is set to stop the loop
        if exit_flag:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using MTCNN
        detections = detector.detect_faces(frame)

        current_faces = set()  # Track current detected faces
        for result in detections:
            x, y, width, height = result['box']
            x, y, width, height = max(0, x), max(0, y), min(width, frame.shape[1]), min(height, frame.shape[0])

            # Extract face region
            face = frame[y:y + height, x:x + width]
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Recognize the face
            label, confidence = recognizer.predict(gray_face)

            if confidence < 100:  # Assuming a threshold for recognition confidence
                person_name = names.get(str(label), "Unknown")
                current_faces.add(person_name)

                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Track people entering and exiting based on detected faces
        if camera_id == 0:
            detected_faces = detected_faces_cam0
            people_in = people_in_cam0
            opposite_people_out = people_out_cam1  # Corresponding people out in Camera 1
        else:
            detected_faces = detected_faces_cam1
            people_out = people_out_cam1
            opposite_people_in = people_in_cam0  # Corresponding people in from Camera 0

        for person in current_faces:
            if person not in detected_faces:
                detected_faces[person] = y  # Save the y position of the person

        for person in list(detected_faces.keys()):
            if person not in current_faces:  # If person is no longer detected
                if camera_id == 0:
                    people_in += 1  # Count as entered in Camera 0
                    if opposite_people_out > 0:
                        opposite_people_out -= 1  # Decrease people out in Camera 1
                    print(f"{person} entered (Camera {camera_id})")  # Debug print
                else:
                    people_out += 1  # Count as exited in Camera 1
                    if opposite_people_in > 0:
                        opposite_people_in -= 1  # Decrease people in from Camera 0
                    print(f"{person} exited (Camera {camera_id})")  # Debug print
                del detected_faces[person]  # Remove after counting

        # Update global variables
        if camera_id == 0:
            people_in_cam0 = people_in
            people_out_cam1 = opposite_people_out  # Sync Camera 1 people out
        else:
            people_out_cam1 = people_out
            people_in_cam0 = opposite_people_in  # Sync Camera 0 people in

        # Display the results
        if camera_id == 0:
            cv2.putText(frame, f'People In Camera 0: {people_in_cam0}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'People Out Camera 1: {people_out_cam1}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f'People Out Camera 1: {people_out_cam1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'People In Camera 0: {people_in_cam0}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(f'People Counting Camera {camera_id}', frame)

        # Check for 'q' key press to stop program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Create two threads for each camera
camera_1_thread = threading.Thread(target=process_camera, args=(0,))
camera_2_thread = threading.Thread(target=process_camera, args=(1,))

# Start the threads
camera_1_thread.start()
camera_2_thread.start()

# Wait for both threads to finish
camera_1_thread.join()
camera_2_thread.join()
