import cv2
import time
import threading
import queue
from collections import deque
from functools import partial

class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.Q = queue.Queue(maxsize=32)

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                if grabbed:
                    self.Q.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True
        self.stream.release()