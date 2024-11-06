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
people_in_cam0 = 0  # Count for people who entered
people_out_cam1 = 0  # Count for people who exited
detected_faces_cam0 = {}
detected_faces_cam1 = {}

# Flag to stop threads
exit_flag = False

# Record of counted people to avoid double counting
counted_people_in = set()
counted_people_out = set()

# Function for processing a single camera   
def process_camera(camera_id):
    global people_in_cam0, people_out_cam1, detected_faces_cam0, detected_faces_cam1, exit_flag, counted_people_in, counted_people_out

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

            # Check if recognized as a known person
            person_name = names.get(str(label), "Unknown")
            if person_name != "Unknown" and confidence < 100:  # Only process known names
                current_faces.add(person_name)

                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Track people entering and exiting based on detected faces
        if camera_id == 0:
            detected_faces = detected_faces_cam0
            counted_people_set = counted_people_in
        else:
            detected_faces = detected_faces_cam1
            counted_people_set = counted_people_out

        for person in current_faces:
            if person not in detected_faces:
                detected_faces[person] = y  # Save the y position of the person

        for person in list(detected_faces.keys()):
            if person not in current_faces:  # If person is no longer detected
                # Process as entry for Camera 0 (in)
                if camera_id == 0 and person not in counted_people_in and person not in counted_people_out:
                    people_in_cam0 += 1  # Count as entered in Camera 0
                    counted_people_in.add(person)
                    print(f"{person} entered (Camera {camera_id})")  # Debug print
                
                # Process as exit for Camera 1 (out) with decrement in camera in count
                elif camera_id == 1 and person in counted_people_in:
                    people_in_cam0 -= 1  # Decrease count from in camera
                    counted_people_in.remove(person)  # Remove from entry list, enabling re-entry in the future
                    counted_people_out.add(person)    # Mark as exited
                    print(f"{person} exited and reset for re-entry (Camera {camera_id})")  # Debug print

                del detected_faces[person]  # Remove after counting

        # Display the results on both cameras
        cv2.putText(frame, f'People In (Camera 0): {people_in_cam0}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'People Out (Camera 1): {len(counted_people_out)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
