import cv2
import time
import csv
from datetime import datetime
from pathlib import Path


# --------------------------------------------------
# Settings
# --------------------------------------------------

CAMERA_DEVICE = "/dev/video0"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 800
CAMERA_FPS = 30

CAPTURE_INTERVAL = 1.0

# Save FS_IMG inside the Raspberry Pi user's home folder
MAIN_FOLDER = Path.home() / "FS_IMG"


# --------------------------------------------------
# Create the next experiment folder
# --------------------------------------------------

def create_experiment_folder():
    MAIN_FOLDER.mkdir(parents=True, exist_ok=True)

    experiment_number = 1

    while True:
        folder_name = f"exp{experiment_number:05d}"
        experiment_folder = MAIN_FOLDER / folder_name

        if not experiment_folder.exists():
            experiment_folder.mkdir()
            return experiment_folder

        experiment_number += 1


# --------------------------------------------------
# Main program
# --------------------------------------------------

experiment_folder = create_experiment_folder()

print("Images will be saved in:")
print(experiment_folder.resolve())


# --------------------------------------------------
# Create timestamp CSV
# --------------------------------------------------

csv_path = experiment_folder / "timestamps.csv"

with open(csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["image_name", "time_captured"])


# --------------------------------------------------
# Open camera
# --------------------------------------------------

cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera device {CAMERA_DEVICE}."
    )


# MJPG often allows USB cameras to use higher resolutions
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)


# Read back what the camera actually accepted
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print()
print(f"Requested resolution: {CAMERA_WIDTH} x {CAMERA_HEIGHT}")
print(f"Actual resolution:    {actual_width} x {actual_height}")
print(f"Camera FPS:           {actual_fps}")
print()
print("Capturing one image per second.")
print("Press q in the preview window to stop.")


# --------------------------------------------------
# Capture loop
# --------------------------------------------------

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

        image_name = f"image{image_number:05d}.png"
        image_path = experiment_folder / image_name

        saved = cv2.imwrite(str(image_path), frame)

        if saved:

            # Record the time immediately after the image is saved
            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # Append image name and timestamp to CSV
            with open(csv_path, "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    image_name,
                    timestamp
                ])

            print(
                f"Saved {image_path} "
                f"({frame.shape[1]} x {frame.shape[0]}) "
                f"at {timestamp}"
            )

            image_number += 1

        else:
            print(f"ERROR: Could not save {image_path}")

        next_capture_time += CAPTURE_INTERVAL

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Image capture stopped.")
print(f"Timestamps saved to: {csv_path}")