# ==============================================================================
# realtime_pipeline.py
#
# Author: Aryaman Jaiswal 
#
# Description:
# This script serves as the primary real-time data processing engine for the
# sign language translator. It performs several critical tasks in a continuous
# loop:
#   1. Captures live video from the default webcam.
#   2. Uses the MediaPipe library to detect and track hand landmarks.
#   3. Implements a vision-based heuristic (wrist velocity) to identify and
#      capture "key postures," replacing the need for a Kinect sensor.
#   4. Manages a buffer of these key frames.
#   5. Detects when a sign is complete (when the hand leaves the frame) and
#      triggers the generation of an Accumulative Video Motion (AVM) image.
#   6. Provides real-time visual feedback by drawing landmarks and status text.
#
# This script was the precursor to the final GUI application and contains all
# the core logic that was later integrated into the application's worker thread.
# ==============================================================================

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- Module Imports ---
# We import the generate_avm function from our other module. 
from avm_generator import generate_avm

# --- 1. Initialization of Core Components ---
# Initialize the MediaPipe Hands solution. This loads the pre-trained ML model
# for hand detection and tracking.
mp_hands = mp.solutions.hands
# Initialize the MediaPipe drawing utility, which provides helper functions
# to draw the hand skeleton (landmarks and connections) on the image.
mp_drawing = mp.solutions.drawing_utils

# --- 2. Configuration and State Variables ---
# A 'deque' (double-ended queue) is used for the buffer. It's like a list but
# has a fixed maximum size (maxlen). When a new item is added and the deque is
# full, the oldest item is automatically discarded. This is perfect for keeping
# a rolling buffer of the most recent N frames.
KEY_FRAMES_BUFFER_SIZE = 16 
key_frames_buffer = deque(maxlen=KEY_FRAMES_BUFFER_SIZE)

# This threshold is the core of our key posture detection heuristic. If the calculated
# speed of the wrist between two frames is less than this value, we consider the
# hand to be "paused," and we capture the frame. This value may need tuning for
# different lighting conditions or camera frame rates.
WRIST_VELOCITY_THRESHOLD = 0.01

# These variables store information between loop iterations (frames). They are
# essential for calculating changes over time, like velocity.
previous_wrist_coords = None # Stores the wrist (x,y) from the previous frame.
is_hand_paused = False       # A flag to prevent capturing hundreds of frames during one long pause.

# --- 3. Video Capture Setup ---
# cv2.VideoCapture(0) opens a connection to the default webcam. If you have
# multiple cameras, the index might be 1, 2, etc.
cap = cv2.VideoCapture(0)

# --- 4. MediaPipe Model Context ---
# The 'with' statement is a best practice for managing resources. It ensures that
# the MediaPipe Hands model is properly initialized and, more importantly, that
# its resources are cleaned up automatically when the block is exited.
with mp_hands.Hands(
    min_detection_confidence=0.7, # The model must be at least 70% confident it sees a hand.
    min_tracking_confidence=0.5) as hands: # Once a hand is found, it can be tracked with 50% confidence.

    # --- 5. The Main Real-Time Loop ---
    # This loop runs continuously, processing one frame from the webcam at a time.
    while cap.isOpened():
        # cap.read() returns two values: a boolean 'success' flag (True if a frame
        # was read correctly) and the 'image' itself as a NumPy array.
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue # Skips the rest of the loop and tries to read the next frame.

        # Make a clean, unmodified copy of the frame. This is crucial because we
        # will later draw landmarks and text on the 'image' variable for display.
        # We need to store the original, clean frame in our buffer for the models.
        original_image_for_buffer = image.copy()

        # --- 6. Image Processing for MediaPipe ---
        # For performance, we tell NumPy the image is not writeable. This allows
        # MediaPipe to process the image data more efficiently (pass-by-reference).
        image.flags.writeable = False
        # MediaPipe's models were trained on RGB images, but OpenCV reads frames
        # in BGR format. We must convert the color space for the model to work correctly.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # This is the core ML inference step for hand tracking. The .process()
        # method runs the MediaPipe model on the image.
        results = hands.process(image_rgb)
        
        # We now make the image writeable again so we can draw on it.
        image.flags.writeable = True
        # Convert back to BGR for correct color display with OpenCV's imshow().
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # --- 7. Key Posture and AVM Trigger Logic ---
        # 'results.multi_hand_landmarks' will contain data if a hand was found,
        # otherwise it will be None.
        if results.multi_hand_landmarks:
            # For simplicity, we only process the first hand detected.
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw the hand skeleton on the display image.
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract the coordinates of the wrist (landmark #0). MediaPipe
            # returns normalized coordinates (0.0 to 1.0), so they are
            # independent of the video resolution.
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            current_wrist_coords = np.array([wrist.x, wrist.y])

            # We can only calculate velocity if we have a position from the previous frame.
            if previous_wrist_coords is not None:
                # Calculate the Euclidean distance between the current and previous
                # wrist positions. This simple distance is our proxy for speed/velocity.
                current_velocity = np.linalg.norm(current_wrist_coords - previous_wrist_coords)
                
                # Check if the hand has paused.
                if current_velocity < WRIST_VELOCITY_THRESHOLD:
                    # We only capture the *first* frame of a pause. The 'is_hand_paused'
                    # flag ensures that we don't keep adding frames if the user
                    # holds their hand still for a long time.
                    if not is_hand_paused:
                        print(f"Key Posture Captured! Velocity: {current_velocity:.4f}")
                        # Add the clean, original frame to our buffer.
                        key_frames_buffer.append(original_image_for_buffer)
                        is_hand_paused = True # Set the flag to true for this pause.
                else:
                    is_hand_paused = False # If moving, reset the flag.

            # Store the current position to be used in the next frame's calculation.
            previous_wrist_coords = current_wrist_coords
        else:
            # This block executes if NO hand is detected in the frame.
            # This is our trigger for processing a completed sign.
            if len(key_frames_buffer) > 0:
                print("Hand removed. Processing captured frames to generate AVM...")
                
                # Convert the deque buffer to a standard list for our function.
                frames_to_process = list(key_frames_buffer)
                
                # Call our AVM generator with the captured frames.
                _, _, bidirectional_avm = generate_avm(frames_to_process)
                
                # If the AVM was created successfully, display it in a new window.
                if bidirectional_avm is not None:
                    cv2.imshow("Generated AVM", bidirectional_avm)
                
                # Clear the buffer to be ready for the next sign.
                key_frames_buffer.clear()
                print("Buffer cleared. Ready for next sign.")
            
            # Reset state variables since the hand is gone.
            previous_wrist_coords = None
            is_hand_paused = False

        # --- 8. Visual Feedback and Display ---
        # Flip the image horizontally to create an intuitive "mirror" view.
        image = cv2.flip(image, 1)

        # Draw status text on the flipped image. The text will now be readable.
        cv2.putText(image, f"Wrist Velocity: {current_velocity:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Key Frames Captured: {len(key_frames_buffer)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if is_hand_paused:
            cv2.putText(image, "KEY POSTURE CAPTURED", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the final, annotated image.
        cv2.imshow('Key Posture Detection', image)

        # --- 9. Exit Condition ---
        # cv2.waitKey(5) waits for 5ms for a key press.
        # '& 0xFF == 27' checks if the key pressed was the 'Esc' key.
        if cv2.waitKey(5) & 0xFF == 27:
            break

# --- 10. Cleanup ---
# Release the webcam resource.
cap.release()
# Close all OpenCV windows.
cv2.destroyAllWindows()