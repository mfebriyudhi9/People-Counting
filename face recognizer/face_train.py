import cv2
import numpy as np
from PIL import Image
import os
from mtcnn import MTCNN
import json

def get_images_and_labels(path):
    detector = MTCNN()
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('jpg', 'jpeg', 'png'))]
    face_samples = []
    ids = []

    for image_path in image_paths:
        try:
            PIL_img = Image.open(image_path).convert('RGB')
            img_numpy = np.array(PIL_img)

            results = detector.detect_faces(img_numpy)
            if not results:
                continue  # Skip images with no detected faces

            for result in results:
                x, y, w, h = result['box']
                face = img_numpy[y:y+h, x:x+w]
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_samples.append(gray_face)
                
                try:
                    id = int(os.path.split(image_path)[-1].split("-")[1].split(".")[0])  # Adjust ID extraction as needed
                    ids.append(id)
                except ValueError:
                    print(f"[WARNING] ID extraction failed for file: {image_path}")
                    continue

        except Exception as e:
            print(f"[ERROR] Error processing file {image_path}: {e}")

    return face_samples, ids

def load_names(filename):
    if not os.path.exists(filename):
        print(f"[ERROR] The file {filename} does not exist.")
        return {}
    with open(filename, 'r') as f:
        return json.load(f)

def recognize_faces(image_path, recognizer, names):
    detector = MTCNN()
    img = cv2.imread(image_path)
    results = detector.detect_faces(img)
    
    recognized_faces = []
    for result in results:
        x, y, w, h = result['box']
        face = img[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        label, confidence = recognizer.predict(gray_face)

        if confidence < 100:  # Adjust threshold as necessary
            name = names.get(str(label), "unknown")
        else:
            name = "unknown"
        
        recognized_faces.append((name, (x, y, w, h)))

    return recognized_faces

if __name__ == "__main__":
    training_path = './images_muka/'
    
    if not os.path.exists(training_path):
        print(f"[ERROR] The directory {training_path} does not exist.")
        exit()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("\n[INFO] Training...")

    faces, ids = get_images_and_labels(training_path)
    
    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))
        recognizer.write('trainer.yml')
        print(f"\n[INFO] {len(np.unique(ids))} faces trained. Exiting Program")
    else:
        print("\n[ERROR] No faces found. Exiting Program")
        exit()