import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound  # Import playsound for alarm

playsound('alarm.wav')

# Initialize Mediapipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up webcam feed
cap = cv2.VideoCapture(0)

# Define colors for lines
BLUE = (255, 0, 0)  # Safety Border Line
RED = (0, 0, 255)   # Danger Line
GREEN = (0, 255, 0)  # Text color

# Define line positions (you can adjust these)
SAFETY_LINE_Y = 100
DANGER_LINE_Y = 50

# Alarm flag to ensure alarm sounds only once per crossing
alarm_triggered = False

# Initialize hands detection
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image and find hands
        results = hands.process(image)
        
        # Draw the lines on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.line(image, (0, SAFETY_LINE_Y), (image.shape[1], SAFETY_LINE_Y), BLUE, 2)
        cv2.line(image, (0, DANGER_LINE_Y), (image.shape[1], DANGER_LINE_Y), RED, 2)

        # Check if any hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Check the position of the index finger tip (landmark 8)
                index_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]
                
                # Check if the finger is below the danger line
                if index_finger_tip_y < DANGER_LINE_Y and not alarm_triggered:
                    cv2.putText(image, "Danger! System Shutting Down!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2, cv2.LINE_AA)
                    print("Index Finger Touched the line, shutting down the system")
                    playsound('alarm.wav')  # Play alarm sound
                    alarm_triggered = True  # Set the alarm flag
                    
                elif index_finger_tip_y >= DANGER_LINE_Y and index_finger_tip_y < SAFETY_LINE_Y:
                    cv2.putText(image, "Warning! Finger near danger zone!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA)
                    print("Finger near danger zone")
                    alarm_triggered = False  # Reset the alarm flag when finger is out of danger

        # Display FPS
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cv2.putText(image, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA)

        # Show the image with landmarks
        cv2.imshow('SafetyCAM', image)

        # Break loop with 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
