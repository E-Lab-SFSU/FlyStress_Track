import cv2
import os
import time

SAVE_FOLDER = "FS_IMG"
CAPTURE_INTERVAL = 1.0

os.makedirs(SAVE_FOLDER, exist_ok=True)

# Windows:
CAMERA_INDEX = 1
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# Raspberry Pi alternative:
# cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open the selected camera.")

ret, frame = cap.read()

if not ret:
    cap.release()
    raise RuntimeError(
        "The camera opened, but OpenCV could not retrieve an image."
    )

height, width = frame.shape[:2]

print(f"Camera opened successfully.")
print(f"Image resolution: {width} x {height}")
print("Press q to stop.")

image_number = 1
next_capture_time = time.monotonic()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to retrieve an image from the camera.")
        break

    cv2.imshow("Camera Preview", frame)

    current_time = time.monotonic()

    if current_time >= next_capture_time:
        filename = f"image{image_number:05d}.png"
        filepath = os.path.join(SAVE_FOLDER, filename)

        if cv2.imwrite(filepath, frame):
            print(f"Saved: {filepath}")
            image_number += 1
        else:
            print(f"Could not save: {filepath}")

        next_capture_time += CAPTURE_INTERVAL

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()