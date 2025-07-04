import cv2
import os
import time

os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"

image_paths = [
    "output_images/Video_1.bmp"
]

while True:
    for i, path in enumerate(image_paths):

        try:
            img = cv2.imread(path)
            if img is None:
                continue
        except:
            continue

        try:
            cv2.imshow(f"output {i}", img)
        except:
            continue

        key = cv2.waitKey(1)
        if key == 27 or key in [ord('q'), ord('Q')]:
            cv2.destroyAllWindows()
            exit()