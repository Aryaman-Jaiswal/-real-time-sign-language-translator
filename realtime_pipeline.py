import cv2
import mediapipe as mp
import numpy as np

# --- 1. Initialize MediaPipe Hands and Drawing Utilities ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

        # --- 5. Process the Image ---
        # To improve performance, mark the image as not writeable.
        # THIS IS THE CORRECTED LINE:
        image.flags.writeable = False
        
        # Convert the color space from BGR to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hand landmarks.
        results = hands.process(image)
        
        # --- 6. Draw the Annotations on the Image ---
        # Now make the image writeable again.
        # THIS IS THE SECOND CORRECTED LINE:
        image.flags.writeable = True

        # Convert the color space back from RGB to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if any hands were detected.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

        # --- 7. Display the final image ---
        cv2.imshow('MediaPipe Hand Tracking', cv2.flip(image, 1))

        # --- 8. Exit Condition ---
        if cv2.waitKey(5) & 0xFF == 27:
            break

# --- 9. Cleanup ---
cap.release()
cv2.destroyAllWindows()