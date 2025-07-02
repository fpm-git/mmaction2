import cv2
import os 
import time

# image_paths = ['../micron_output_images/Video_1.jpg',
#                '../micron_output_images/Video_2.jpg',
#                '../micron_output_images/Video_3.jpg',
#                '../micron_output_images/Video_4.jpg',
# ]
image_paths = ['./output_images/Video_3.jpg']

while True:
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        # img = cv2.resize(img, (640, 480))

        if img is not None:
            cv2.imshow(f"test_output_{i}", img)
        else:
            print("img is None")
        
        key = cv2.waitKey(1)
        if key == 27 or key in [ord('q'), ord('Q')]:
            cv2.destroyAllWindows()
            exit()

    # time.sleep(0.05)
