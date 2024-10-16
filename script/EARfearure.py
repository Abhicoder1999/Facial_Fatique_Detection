import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# EAR Calculation function
def calculate_ear(eye_landmarks):
    # Vertical distances (p2-p6 and p3-p5)
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))

    # Horizontal distance (p1-p4)
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))

    # Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Specific eye landmarks from MediaPipe Face Mesh (left and right eye indices)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]  # Left eye
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]  # Right eye

# Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the image color (BGR to RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect face landmarks
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape  # Get image shape
            
            # Extract the landmark points for both eyes
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE_LANDMARKS]
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE_LANDMARKS]

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            # Display EAR for both eyes on the image
            cv2.putText(frame, f'Left EAR: {left_ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Right EAR: {right_ear:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Optionally, draw eye landmarks
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 255), -1)

    # Show the result in a window
    cv2.imshow('Eye Aspect Ratio (EAR)', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows() 