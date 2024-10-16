import numpy as np
import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)


FACE_POSE_LANDMARKS = [1, 33, 263, 61, 291, 199] # Nose mouth ears head chin ( Apart from nose order not sure)
LEFT_EYE_LANDMARKS  = [362, 385, 387, 263, 373, 380]  # Left eye
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]  # Right eye
MOUTH_LANDMARKS     = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]  # Upper & lower lip points
                    #   0    1   2    3   4   5   6    7    8   9     10   11   12   13   14  15  16  17  18  19                                 


# MAR Calculation function
def calculate_mar(mouth_landmarks):
    # Vertical distances (p14-p18 and p15-p17)
    A = np.linalg.norm(np.array(mouth_landmarks[16]) - np.array(mouth_landmarks[4]))  # p37-p84
    B = np.linalg.norm(np.array(mouth_landmarks[14]) - np.array(mouth_landmarks[6]))  # p267-p314

    # Horizontal distance (p13-p19)
    C = np.linalg.norm(np.array(mouth_landmarks[0]) - np.array(mouth_landmarks[10]))  # p61-p291

    # Mouth Aspect Ratio
    mar = (A + B) / (2.0 * C)
    return mar


def calculate_ear(eye_landmarks):
    # Vertical distances (p2-p6 and p3-p5)
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))

    # Horizontal distance (p1-p4)
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))

    # Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)
    return ear


def calculate_pose(face_2d, face_3d, img_w, img_h):
    face_2d = np.array(face_2d,dtype=np.float64)
    face_3d = np.array(face_3d,dtype=np.float64)

    focal_length = 1 * img_w

    cam_matrix = np.array([[focal_length,0,img_h/2],
                                  [0,focal_length,img_w/2],
                                  [0,0,1]])
    
    print("cam_matrix:", cam_matrix)

    distortion_matrix = np.zeros((4,1),dtype=np.float64)

    success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)


    #getting rotational of face
    rmat,jac = cv2.Rodrigues(rotation_vec)

    angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    print("x y z:", x, y, z)

    return [x,y,z]


cap = cv2.VideoCapture(0)
while cap.isOpened():

    success, image = cap.read()
    if not success:
        break

    start = time.time()
    image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB) #flipped for selfie view
    image.flags.writeable = False
    results = face_mesh.process(image)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
    h, w, _ = image.shape  # Get image shape
    face_pose_2d = []
    face_pose_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                    if idx ==1:
                        nose_2d = (lm.x * w,lm.y * h)
                        nose_3d = (lm.x * w,lm.y * h,lm.z * 3000)
                    x,y = int(lm.x * w),int(lm.y * h)

                    face_pose_2d.append([x,y])
                    face_pose_3d.append(([x,y,lm.z]))
            
            # TO DO convert to single line 
            # face_pose_2d = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in FACE_POSE_LANDMARKS]
            # face_pose_3d = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h), int(face_landmarks.landmark[i].z )) for i in FACE_POSE_LANDMARKS]
            
            mouth = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in MOUTH_LANDMARKS]
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE_LANDMARKS]
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE_LANDMARKS]


            mar = calculate_mar(mouth)
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            pose = calculate_pose(face_pose_2d,face_pose_3d, w, h)

            
                        # Display MAR on the image
            cv2.putText(image, f'MAR: {mar:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f'Left EAR: {left_ear:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f'Right EAR: {right_ear:.2f}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image,"x: " + str(np.round(pose[0],2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(image,"y: "+ str(np.round(pose[1],2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(image,"z: "+ str(np.round(pose[2], 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            # Optionally, draw eye landmarks   
            for point in left_eye + right_eye:
                cv2.circle(image, point, 2, (255,0 , 255), -1)
            for point in mouth:
                cv2.circle(image, point, 2, (0, 255, 255), -1)
                
    # Show the result in a window
    cv2.imshow('Facial Features', image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
