# People Counting and Face Recognition System

This project is a real-time people counting and face recognition system that utilizes two cameras to track entries and exits. It uses MTCNN for face detection and LBPH (Local Binary Patterns Histograms) for face recognition. This system is designed to keep track of people entering and exiting an area by detecting faces on two separate cameras, designated as "in" and "out" cameras.


## Features

- **Two-camera setup**: Camera 0 serves as the "in" camera for detecting entries, while Camera 1 serves as the "out" camera for detecting exits.
- **Face Recognition**: Uses MTCNN for face detection and LBPH for face recognition.
- **Real-time Counting**: Keeps track of the number of people entering and exiting in real-time.
- **Entry/Exit Condition**: Ensures that each person must exit before being detected again as a new entry.
- **Display**: Shows camera feeds with the number of people counted and names of recognized individuals.


