import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- NEW: Import your AVM generator function ---
from avm_generator import generate_avm

# --- 1. Initialize MediaPipe and Drawing Utilities ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Key Posture Detection Constants and Variables ---
KEY_FRAMES_BUFFER_SIZE = 16 
key_frames_buffer = deque(maxlen=KEY_FRAMES_BUFFER_SIZE)
WRIST_VELOCITY_THRESHOLD = 0.01
previous_wrist_coords = None
is_hand_paused = False

# --- 2. Start Video Capture ---
cap = cv2.VideoCapture(0)

# --- 3. Set up the MediaPipe Hands model ---
with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    # --- 4. The Main Real-Time Loop ---
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        h, w, _ = image.shape
        
        # Make a clean copy of the original image for the buffer
        original_image_for_buffer = image.copy()

        # --- 5. Process the Image with MediaPipe ---
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # --- 6. Draw Annotations and Implement Logic ---
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_velocity = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            current_wrist_coords = np.array([wrist.x, wrist.y])

            if previous_wrist_coords is not None:
                current_velocity = np.linalg.norm(current_wrist_coords - previous_wrist_coords)
                if current_velocity < WRIST_VELOCITY_THRESHOLD:
                    if not is_hand_paused:
                        print(f"Key Posture Captured! Velocity: {current_velocity:.4f}")
                        # Add the clean, UNFlipped, UNAnnotated image to the buffer
                        key_frames_buffer.append(original_image_for_buffer)
                        is_hand_paused = True
                else:
                    is_hand_paused = False
            previous_wrist_coords = current_wrist_coords
        else:
            # --- NEW: AVM Generation Trigger ---
            # If no hand is detected, AND we have frames in our buffer,
            # it means a sign has just finished.
            if len(key_frames_buffer) > 0:
                print("Hand removed. Processing captured frames to generate AVM...")
                
                # Convert deque to a list for the function
                frames_to_process = list(key_frames_buffer)
                
                # Call your AVM generator
                _, _, bidirectional_avm = generate_avm(frames_to_process)
                
                # Display the result if it was successful
                if bidirectional_avm is not None:
                    cv2.imshow("Generated AVM", bidirectional_avm)
                
                # Clear the buffer to be ready for the next sign
                key_frames_buffer.clear()
                print("Buffer cleared. Ready for next sign.")
            
            previous_wrist_coords = None
            is_hand_paused = False

        # --- 7. Flip the image and draw text ---
        image = cv2.flip(image, 1)

        cv2.putText(image, f"Wrist Velocity: {current_velocity:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Key Frames Captured: {len(key_frames_buffer)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if is_hand_paused:
            cv2.putText(image, "KEY POSTURE CAPTURED", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- 8. Display the main feed ---
        cv2.imshow('Key Posture Detection', image)

        # --- 9. Exit Condition ---
        if cv2.waitKey(5) & 0xFF == 27:
            break

# --- 10. Cleanup ---
cap.release()
cv2.destroyAllWindows()