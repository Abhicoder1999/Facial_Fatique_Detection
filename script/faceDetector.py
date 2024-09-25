import dlib

class FaceDetector:
    def __init__(self) -> None:
        self.frontalFaceDetector = dlib.get_frontal_face_detector()

    def run(self,imgRGB,view):
        allFaces = self.frontalFaceDetector(imgRGB, 0)
        if view:
             print("List of all faces detected: ",len(allFaces))
        return allFaces
    