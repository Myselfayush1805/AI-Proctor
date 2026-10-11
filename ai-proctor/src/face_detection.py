import cv2
import time
class FaceDetector:
    def __init__(self):
        self.face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        self.absence_start=None
    def detect(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=self.face_cascade.detectMultiScale(gray,1.3,5)
        return faces 
    def check_absence(self,faces):
        if len(faces)==0:
            if self.absence_start is None:
                self.absence_start=time.time()
            elif time.time()-self.absence_start>5:
                return True
        else:
            self.absence_start=None
        return False