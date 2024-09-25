import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt



class FeatureExtractor():
    def __init__(self) -> None:
        self.ear_values = []
        self.mar_values = []
        self.time_steps = []
        
        Model_PATH = "shape_predictor_68_face_landmarks.dat"
        self.faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

    def run(self,imageRGB, allFaces, view):

        for k in range(0, len(allFaces)):
            # dlib rectangle class will detecting face so that landmark can apply inside of that area
            faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()),int(allFaces[k].top()),
            int(allFaces[k].right()),int(allFaces[k].bottom()))
            
            detectedLandmarks = self.faceLandmarkDetector(imageRGB, faceRectangleDlib)
            
            # count number of landmarks we actually detected on image
            if k==0:
                print("Total number of face landmarks detected ",len(detectedLandmarks.parts()))

            self.facePoints(imageRGB, detectedLandmarks)
            self.getFeatures(imageRGB,detectedLandmarks)

        if view:
            cv2.imshow("Face_Landmarks",imageRGB)
            cv2.waitKey(1)


    # This below mehtod will draw all those points which are from 0 to 67 on face one by one.
    def drawPoints(self,image, faceLandmarks, startpoint, endpoint, isClosed=False):
        points = []
        for i in range(startpoint, endpoint + 1):
            point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
            points.append(point)
            cv2.putText(
                image,
                str(i),
                point,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        points = np.array(points, dtype=np.int32)
        cv2.polylines(
            image, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8
        )


    # Use this function for 70-points facial landmark detector model
    # we are checking if points are exactly equal to 68, then we draw all those points on face one by one
    def facePoints(self,image, faceLandmarks):

        assert faceLandmarks.num_parts == 68
        self.drawPoints(image, faceLandmarks, 0, 16)  # Jaw line
        self.drawPoints(image, faceLandmarks, 17, 21)  # Left eyebrow
        self.drawPoints(image, faceLandmarks, 22, 26)  # Right eyebrow
        self.drawPoints(image, faceLandmarks, 27, 30)  # Nose bridge
        self.drawPoints(image, faceLandmarks, 30, 35, True)  # Lower nose
        self.drawPoints(image, faceLandmarks, 36, 41, True)  # Left eye
        self.drawPoints(image, faceLandmarks, 42, 47, True)  # Right Eye
        self.drawPoints(image, faceLandmarks, 48, 59, True)  # Outer lip
        self.drawPoints(image, faceLandmarks, 60, 67, True)  # Inner lip


    # Use this function for any model other than
    # 70 points facial_landmark detector model
    def facePoints2(image, faceLandmarks, color=(0, 255, 0), radius=4):
        for p in faceLandmarks.parts():
            cv2.circle(image, (p.x, p.y), radius, color, -1)


    def calculateEAR(self, eye):
        # Vertical distances
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        # Horizontal distance
        C = np.linalg.norm(eye[0] - eye[3])
        # EAR calculation
        ear = (A + B) / (2.0 * C)
        return ear


    def getLeftEAR(self,faceLandmarks):
        assert faceLandmarks.num_parts == 68
        left_eye = np.array(
            [
                (faceLandmarks.part(36).x, faceLandmarks.part(36).y),
                (faceLandmarks.part(37).x, faceLandmarks.part(37).y),
                (faceLandmarks.part(38).x, faceLandmarks.part(38).y),
                (faceLandmarks.part(39).x, faceLandmarks.part(39).y),
                (faceLandmarks.part(40).x, faceLandmarks.part(40).y),
                (faceLandmarks.part(41).x, faceLandmarks.part(41).y),
            ],
            np.float32,
        )

        return self.calculateEAR(left_eye)


    def getRightEAR(self,faceLandmarks):
        assert faceLandmarks.num_parts == 68
        right_eye = np.array(
            [
                (faceLandmarks.part(42).x, faceLandmarks.part(42).y),
                (faceLandmarks.part(43).x, faceLandmarks.part(43).y),
                (faceLandmarks.part(44).x, faceLandmarks.part(44).y),
                (faceLandmarks.part(45).x, faceLandmarks.part(45).y),
                (faceLandmarks.part(46).x, faceLandmarks.part(46).y),
                (faceLandmarks.part(47).x, faceLandmarks.part(47).y),
            ],
            np.float32,
        )

        return self.calculateEAR(right_eye)


    def getMAR(self,faceLandmarks):

        mouth = np.array(
            [
                (faceLandmarks.part(48).x, faceLandmarks.part(48).y),
                (faceLandmarks.part(50).x, faceLandmarks.part(50).y),
                (faceLandmarks.part(52).x, faceLandmarks.part(52).y),
                (faceLandmarks.part(54).x, faceLandmarks.part(54).y),
                (faceLandmarks.part(56).x, faceLandmarks.part(56).y),
                (faceLandmarks.part(58).x, faceLandmarks.part(58).y),
            ]
        )

        # Vertical distances
        A = np.linalg.norm(mouth[1] - mouth[5])  # ||P50 - P58||
        B = np.linalg.norm(mouth[2] - mouth[4])  # ||P52 - P56||
        # Horizontal distance
        D = np.linalg.norm(mouth[0] - mouth[3])  # ||P48 - P54||
        # MAR calculation
        mar = (A + B) / (2.0 * D)
        return mar


    def getFeatures(self,image, faceLandmarks):
        right_ear = self.getRightEAR(faceLandmarks)
        left_ear = self.getLeftEAR(faceLandmarks)

        mar = self.getMAR(faceLandmarks)
        ear = (left_ear + right_ear) / 2.0

        self.ear_values.append(ear)
        self.mar_values.append(mar)

        # Add a new time step
        self.time_steps.append(len(self.ear_values))

        # Keep the plot within a range of 100 points
        if len(self.time_steps) > 100:
            plt.plot(self.ear_values, self.time_steps)
            plt.plot(self.mar_values, self.time_steps)
            self.ear_values.pop(0)
            self.mar_values.pop(0)
            self.time_steps.pop(0)

        cv2.putText(
            image, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            image, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2
        )

        # Blink detection or other logic can be added based on EAR threshold
        if ear < 0.15:  # This is an arbitrary threshold
            cv2.putText(
                image,
                "BLINK DETECTED!",
                (1000, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        
        if mar > 0.55:  # This is an arbitrary threshold
            cv2.putText(
                image,
                "YAWN DETECTED!",
                (1000, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        