import cv2
from mmaction.utils.stream_source import VideoStream

import os

capture = cv2.VideoCapture("test_videos/download_test_1.mp4")

while True:
    ret, frame = capture.read()
    if not ret:
        print("[Exiting] No more frames to read from")
        break
    cv2.imshow(f"test", frame)
    if cv2.waitKey(1) == ord('q'):
        break