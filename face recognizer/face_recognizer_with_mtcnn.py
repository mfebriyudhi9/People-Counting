import cv2
import json
from mtcnn import MTCNN
import os

def get_names(filename):
    """Load names from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as fs:
            return json.load(fs)
    print(f"[ERROR] {filename} does not exist.")
    return {}

if __name__ == "__main__":
    # Initialize the face recognizer and load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_path = 'trainer.yml'
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file {model_path} does not exist.")
        exit()

    recognizer.read(model_path)

    # Initialize MTCNN for face detection
    detector = MTCNN()
    names = get_names('names.json')

    # Open a connection to the default camera (camera index 0)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Camera not found or cannot be opened.")
        exit()

    cam.set(3, 640)  # Set width
    cam.set(4, 480)  # Set height

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Failed to capture image")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces using MTCNN
        results = detector.detect_faces(img)

        if results:
            for result in results:
                x, y, w, h = result['box']
                face = gray[y:y+h, x:x+w]

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Recognize the face using the trained model
                id, confidence = recognizer.predict(face)

                # Check if the confidence is below the threshold for known faces
                if confidence < 100:  # This threshold can be adjusted
                    name = names.get(str(id), "Unknown")  # Get name from names.json or set as "Unknown"
                    confidence = f"{100 - confidence:.2f}%"
                else:
                    name = "Unknown"  # Set name to "Unknown" if confidence is high
                    confidence = "N/A"

                cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        # Break loop if 'Esc' key is pressed
        if cv2.waitKey(10) & 0xff == 27:
            break

    print("\n[INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()
import cv2
import json
from mtcnn import MTCNN
import os

def get_names(filename):
    """Load names from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as fs:
            return json.load(fs)
    print(f"[ERROR] {filename} does not exist.")
    return {}

if __name__ == "__main__":
    # Initialize the face recognizer and load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_path = 'trainer.yml'
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file {model_path} does not exist.")
        exit()

    recognizer.read(model_path)

    # Initialize MTCNN for face detection
    detector = MTCNN()
    names = get_names('names.json')

    # Open a connection to the default camera (camera index 0)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Camera not found or cannot be opened.")
        exit()

    cam.set(3, 640)  # Set width
    cam.set(4, 480)  # Set height

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Failed to capture image")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces using MTCNN
        results = detector.detect_faces(img)

        if results:
            for result in results:
                x, y, w, h = result['box']
                face = gray[y:y+h, x:x+w]

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Recognize the face using the trained model
                id, confidence = recognizer.predict(face)

                # Check if the confidence is below the threshold for known faces
                if confidence < 100:  # This threshold can be adjusted
                    name = names.get(str(id), "Unknown")  # Get name from names.json or set as "Unknown"
                    confidence = f"{100 - confidence:.2f}%"
                else:
                    name = "Unknown"  # Set name to "Unknown" if confidence is high
                    confidence = "N/A"

                cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        # Break loop if 'Esc' key is pressed
        if cv2.waitKey(10) & 0xff == 27:
            break

    print("\n[INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()