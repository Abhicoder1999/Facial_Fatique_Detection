import cv2
from realsenseStream import Camera
from faceDetector import FaceDetector
from featureExtractor import FeatureExtractor

class fatique_detector:

    def __init__(self) -> None:
        self.camera = Camera()
        self.facedet = FaceDetector()
        self.featext = FeatureExtractor()

    def start(self):
        while True:
            img = self.camera.run(True)
            allFaces = self.facedet.run(img,True)
            self.featext.run(img,allFaces,True)


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def __del__(self):
        self.camera.__del__()


if __name__ == "__main__":
    fd = fatique_detector()
    fd.start()

# Initialize the RealSense pipeline
