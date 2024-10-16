import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

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

# Mouth landmarks from MediaPipe Face Mesh (indices for lower and upper lips)
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]  # Upper & lower lip points
                #   0    1   2    3   4   5   6    7    8   9     10   11   12   13   14  15  16  17  18  19                                 
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
            
            # Extract the landmark points for the mouth
            mouth = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in MOUTH_LANDMARKS]

            # Calculate MAR for the mouth
            mar = calculate_mar(mouth)

            # Display MAR on the image
            cv2.putText(frame, f'MAR: {mar:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Optionally, draw mouth landmarks
            for point in mouth:
                cv2.circle(frame, point, 2, (0, 255, 255), -1)

    # Show the result in a window
    cv2.imshow('Mouth Aspect Ratio (MAR)', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
