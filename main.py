from ultralytics import YOLO
import cv2
import cvzone
import math
import threading
from pygame import mixer  # Use pygame for non-blocking sound playback

# Uncomment next three lines for webcam
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

# Uncomment next line for recorded video
cap = cv2.VideoCapture("assets/v2.mp4")

model = YOLO("yolov8s.pt")

# List of class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "helicopter"
              ]

alert_playing = False  # Flag to track if alert is already playing
stop_alert = threading.Event()  # Event to signal alert thread to stop

# Initialize pygame mixer for sound playback
mixer.init()
mixer.music.load("assets/alert.wav")  # Load the alert sound

def play_alert():
    global alert_playing
    alert_playing = True
    while not stop_alert.is_set():  # Keep playing until stop_alert is set
        if not mixer.music.get_busy():  # Check if the sound is not already playing
            mixer.music.play()
    mixer.music.stop()  # Stop the sound when exiting
    alert_playing = False

alert_thread = None

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, stream=True, save=True)
    person_detected = False  # Reset person detection flag for each frame

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            # Draw bounding box
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class name
            cls = int(box.cls[0])

            if classNames[cls] == "person":
                person_detected = True

            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1.3, thickness=1)

    if person_detected and not alert_playing:
        if not alert_thread or not alert_thread.is_alive():
            alert_thread = threading.Thread(target=play_alert)
            alert_thread.start()  # Start alert thread

    cv2.imshow("video", frame)

    # Exit on single 'q' key press
    if cv2.waitKey(1) == ord('q'):
        stop_alert.set()  # Signal alert thread to stop
        if alert_thread and alert_thread.is_alive():
            alert_thread.join()  # Wait for alert thread to finish
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
