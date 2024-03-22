#%%
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n-seg.pt')

video_path = "./input_data/train/walking/person01_walking_d1_uncomp.avi"

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

