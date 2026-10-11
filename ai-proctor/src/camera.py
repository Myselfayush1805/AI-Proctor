import cv2
class Camera:
    def __init__(self,cam_id=0):
        self.cap=cv2.VideoCapture(cam_id)
        assert self.cap.isOpened(),"Webcam not accessible"
    def read(self):
        ret,frame=self.cap.read()
        if not ret:
            return None
        return frame
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
    
