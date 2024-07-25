import os
import cv2
import torch
import numpy as np
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
output_video_path = os.path.join(VIDEO_DIR, "basic_function_output_video.mp4")

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

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Draw the polygon around the excluded region
    cv2.polylines(frame, [polygon_vertices], isClosed=True, color=(0, 0, 255), thickness=4)

    # Run YOLO on the frame
    results = model(frame)

    # Initialize person count
    person_count = 0

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

    # Display the person count on the top left of the frame
    cv2.putText(
        frame,
        f"Count: {person_count}",
        (10, 30),
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
