import cv2
import time
from threading import Thread
from collections import deque

class FrameData():
    def __init__(self, frame, src):
        self.frame = frame
        self.src = src
        self.frame_time = time.time() * 1000

class VideoStream(object):
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.thread = Thread(target=self.update, args=())
        self.src = src
        self.thread.daemon = True
        self.stopped = True
        self.err_message = None
        self.frame_data = None

    def start(self):
        self.stopped = False
        self.thread.start()

    def stop(self):
        self.stopped = True 
        self.thread.join()
        self.capture.release()
        cv2.destroyAllWindows()

    def update(self):
        if not self.capture.isOpened():
            self.err_message = f"[Error] Cannot open video source: {self.src}"
            print(self.err_message)
            self.stopped = True
            return
    
        while not self.stopped:
            ret, frame = self.capture.read()
            if not ret:
                print(f"[Exiting] No more frames to read from {self.src}")
                self.stopped = True
                break

            if frame is not None:
                self.frame_data = FrameData(frame, self.src)
            time.sleep(0.02)

        self.capture.release()
        cv2.destroyAllWindows()
        print(f"[Info] VideoStream {self.src} has fully stopped.")
        # cv2.destroyAllWindows()
        # exit(1)

    def get_frame(self):
        if self.stopped:
            print(f"[Info] Video stream {self.src} is stopped. No more frames.")
            return None, None, True  
        
        if self.frame_data is None or self.frame_data.frame is None:
            return None, None, False  

        frame = self.frame_data.frame
        frame_src = self.frame_data.src
        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     self.capture.release()
        #     cv2.destroyAllWindows()
        #     exit(1)
        return frame, frame_src, False  
    
    def get_err_message(self):
        return self.err_message