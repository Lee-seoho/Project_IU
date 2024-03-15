import cv2
import numpy as np
import torch
import face_recognition
import time
import os

# hyper-parameters #############
input_video = 'test2.mp4'
output_dir = "output"
################################

cap = cv2.VideoCapture(input_video)
frame_skip = 5
frame_count = 0
os.makedirs(output_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    
    file_name = f'{time.time():.10f}'.replace(".", '_') + '.jpg'
    cv2.imwrite(os.path.join(output_dir, file_name), frame)

    print(frame_count)
    time.sleep(0.001)