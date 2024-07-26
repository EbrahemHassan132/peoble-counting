import os
import cv2
import torch
import numpy as np
import csv
from datetime import datetime
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov10l.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Get the current working directory
VIDEO_DIR = os.getcwd()

# Define the video file path
input_video_path = os.path.join(VIDEO_DIR, "AI Intern Video Tech Task.mp4")

# Define output video file path
output_video_path = os.path.join(VIDEO_DIR, "extra_features_output_video.mp4")

# Define CSV file paths
cashier_csv_path = os.path.join(VIDEO_DIR, "cashier_monitoring.csv")
people_csv_path = os.path.join(VIDEO_DIR, "people_count.csv")

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Define class mapping
class_mapping = {
    0: "person"
}

# Define the excluded region
excluded_region_top_left = (0, 0)
excluded_region_bottom_right = (350, 350)

# Define the vertices of the polygon that covers the whole frame except the excluded region
polygon_vertices = np.array([
    [excluded_region_bottom_right[0], 0],
    [frame_width, 0],
    [frame_width, frame_height],
    [0, frame_height],
    [0, excluded_region_bottom_right[1]],
    [excluded_region_bottom_right[0], excluded_region_bottom_right[1]],
    [excluded_region_bottom_right[0], 0],
    [excluded_region_bottom_right[0], 0]
], np.int32)

# Timer for counting the duration when no person is in the specified area
no_person_frame_count = 0
no_person_duration = 0
absence_start_time = None

# Flag to enable or disable cashier monitoring
cashier_monitoring_enabled = False

# CSV headers
with open(cashier_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Start Time", "End Time", "Duration (seconds)", "People Count", "Location"])

with open(people_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Number of People", "Location"])

# Function to handle button click events
def toggle_cashier_monitoring(event, x, y, flags, param):
    global cashier_monitoring_enabled
    if event == cv2.EVENT_LBUTTONDOWN:
        button_width = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] + 40
        if 20 <= x <= 20 + button_width and 20 <= y <= 50:
            cashier_monitoring_enabled = not cashier_monitoring_enabled

# Create a named window and set mouse callback
cv2.namedWindow("YOLO Output")
cv2.setMouseCallback("YOLO Output", toggle_cashier_monitoring)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Draw the polygon around the targetted region
    cv2.polylines(frame, [polygon_vertices], isClosed=True, color=(0, 0, 255), thickness=4)

    # Draw the button
    button_text = "Monitor: ON" if cashier_monitoring_enabled else "Monitor: OFF"
    button_color = (0, 255, 0) if cashier_monitoring_enabled else (0, 0, 255)
    text_size = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    button_width = text_size[0][0] + 40
    button_height = text_size[0][1] + 20
    cv2.rectangle(frame, (20, 20), ( button_width, 10 + button_height), button_color, -1)
    cv2.putText(frame, button_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Run YOLO on the frame and Initialize person count
    results = model(frame)
    person_count = 0
    person_in_excluded_area = False

    # Draw bounding boxes and labels on the frame for detections within the polygon
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_predictions = result.boxes.cls

        for i in range(len(boxes)):
            box = boxes[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            confidence = confidences[i].item()
            class_prediction = int(class_predictions[i].item())

            # Calculate the center of mass for the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check if the detected object is within the polygon and is a "person"
            if not (excluded_region_top_left[0] <= center_x <= excluded_region_bottom_right[0] and 
                    excluded_region_top_left[1] <= center_y <= excluded_region_bottom_right[1]):
                class_name = class_mapping.get(class_prediction, f"UnknownClass_{class_prediction}")

                if class_name == "person":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f"{class_name}, {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )
                    person_count += 1

            # Check if a person is detected in the excluded region
            if (excluded_region_top_left[0] <= center_x <= excluded_region_bottom_right[0] and
                excluded_region_top_left[1] <= center_y <= excluded_region_bottom_right[1]):
                person_in_excluded_area = True

    # Update the frame count for the duration when no person is in the excluded area
    current_time = datetime.now()
    if person_in_excluded_area:
        no_person_frame_count = 0
        if absence_start_time:
            # Record absence end time and duration
            absence_end_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            absence_duration = no_person_duration
            if absence_duration >= 5:
                with open(cashier_csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([absence_start_time, absence_end_time, round(absence_duration, 2), person_count, "main hall"])
                    absence_logged = True
            absence_start_time = None
    else:
        no_person_frame_count += 1
        no_person_duration = no_person_frame_count / fps
        if not absence_start_time:
            # Record absence start time
            absence_start_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Log the people count every 5 seconds
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % int(fps * 5) == 0:
        with open(people_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([time_str, person_count, "main hall"])

    # Display the person count on the top left of the frame
    cv2.putText(
        frame,
        f"Count: {person_count}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Display the duration of no person in the excluded area on the screen
    if cashier_monitoring_enabled and no_person_duration > 0:
        cv2.putText(
            frame,
            f"Cashier left for: {int(no_person_duration)} seconds",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Write the frame to the output video file
    out.write(frame)

    # Display the frame
    cv2.imshow("YOLO Output", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print("Output video saved to:", output_video_path)
print("Cashier log saved to:", cashier_csv_path)
print("People count log saved to:", people_csv_path)
