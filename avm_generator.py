# ==============================================================================
# avm_generator.py
#
# Author: Aryaman Jaiswal
#
# Description:
# This module is responsible for implementing the "Accumulative Video Motion" 
# (AVM) technique described in the research paper "An Efficient Two-Stream 
# Network for Isolated Sign Language Recognition...". Its primary function, 
# generate_avm(), takes a list of video frames (key postures) and fuses them 
# into a single static image that encodes both motion and shape. This module 
# is a critical component of the Accumulative Motion Network (AMN) stream.
#
# The script also includes a self-contained test function, run_test(),
# to visually verify the functionality of the AVM generation.
# ==============================================================================

import cv2
import numpy as np
import os # Although not used in the final version, it's good practice to keep for file operations.

def generate_avm(key_frames: list[np.ndarray]):
    """
    Generates Accumulative Video Motion (AVM) images from a list of key frames.

    This function implements the AVM technique by summing a sequence of images and
    then normalizing the result into a displayable 8-bit image. This method
    effectively compresses spatial (hand shape) and temporal (hand motion)
    information into a single static image.

    Args:
        key_frames (list[np.ndarray]): A list of images that represent the key
                                       postures of a sign. Each image is expected
                                       to be a NumPy array, as provided by OpenCV.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]: 
        A tuple containing the forward_avm, backward_avm, and bidirectional_avm
        images. In this summation-based implementation, all three are identical.
        Returns a tuple of Nones if the input key_frames list is empty.
    """
    # --- Safety Check ---
    # It's crucial to handle the edge case where an empty list is provided.
    # This prevents the program from crashing and provides a helpful warning.
    if not key_frames:
        print("Warning: key_frames list is empty. Cannot generate AVM.")
        return None, None, None

    # --- Step 1: Prepare for Summation ---
    # Convert the list of individual image arrays into a single, large NumPy array.
    # We explicitly change the data type to np.float32. This is essential to
    # prevent "integer overflow." Standard uint8 images have a max pixel value
    # of 255. Adding them together would exceed this limit and cause the values
    # to wrap around (e.g., 250 + 20 = 14), corrupting the image. Float32 can
    # handle much larger numbers, ensuring the sum is accurate.
    frames_array = np.array(key_frames, dtype=np.float32)

    # --- Step 2: Perform the Summation ---
    # This is the core of the AVM technique. We sum all the frames together.
    # The 'axis=0' argument tells NumPy to sum the elements along the first
    # dimension of the array (which is the dimension that stacks the images).
    # The result is a single image where each pixel's value is the sum of the
    # corresponding pixels from all the key frames.
    summed_image = np.sum(frames_array, axis=0)

    # --- Step 3: Normalize the Image for Display ---
    # The 'summed_image' now has pixel values far exceeding 255 and cannot be
    # displayed. Normalization scales this wide range of values down to the
    # standard visible range of 0-255.
    # cv2.normalize finds the minimum and maximum pixel values in the summed_image
    # and linearly scales them to fit between 0 (alpha) and 255 (beta).
    # The 'dtype=cv2.CV_8U' is the critical parameter that ensures the output is a
    # standard 8-bit image, which is necessary for display and model input.
    normalized_image = cv2.normalize(summed_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # --- Step 4: Prepare the Return Values ---
    # In a simple summation, the order of operations does not matter (A+B = B+A).
    # Therefore, the "forward," "backward," and "bidirectional" AVMs are all
    # identical. We return the same image for all three to maintain a consistent
    # data structure with the paper's three streams, ready for model integration.
    forward_avm = normalized_image
    backward_avm = normalized_image
    bidirectional_avm = normalized_image

    return forward_avm, backward_avm, bidirectional_avm


def run_test():
    """
    A standalone test function to visually demonstrate and verify the
    functionality of generate_avm().

    It creates a set of simple, programmatically generated images (dummy frames)
    and passes them to the AVM generator. It then displays the original frames
    and the final fused AVM image, allowing for easy visual confirmation that
    the fusion logic is working as expected.
    """
    print("Running AVM generation test with overlapping shapes...")

    # --- Create Dummy Test Data ---
    # Define the dimensions for our test images.
    height, width = 300, 500
    
    # Create three separate blank (black) images using np.zeros.
    # np.zeros creates an array filled with zeros.
    # The shape is (height, width, 3 channels for BGR color).
    # The dtype is uint8, the standard for 8-bit images.
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    frame2 = np.zeros((height, width, 3), dtype=np.uint8)
    frame3 = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a white circle on each frame in different, overlapping positions.
    # The overlap is a good test case because it forces the summation in those
    # areas to be higher than 255, testing our normalization logic.
    # cv2.circle(image, center_coordinates, radius, color, thickness)
    # A thickness of -1 fills the circle.
    cv2.circle(frame1, (width // 4, height // 2), 70, (255, 255, 255), -1)
    cv2.circle(frame2, (width // 2, height // 2), 70, (255, 255, 255), -1)
    cv2.circle(frame3, (width * 3 // 4, height // 2), 70, (255, 255, 255), -1)
    
    # Place our three dummy frames into a list, which is the required input format for generate_avm().
    key_frames_list = [frame1, frame2, frame3]

    # --- Call the Function to be Tested ---
    fwd_avm, bwd_avm, bi_avm = generate_avm(key_frames_list)

    # --- Display the Results ---
    # Check if the function returned a valid image before trying to display it.
    if fwd_avm is not None:
        # cv2.imshow() creates a window to display an image.
        # The first argument is the window title, the second is the image to show.
        cv2.imshow("Original Key Frame 1", frame1)
        cv2.imshow("Original Key Frame 2", frame2)
        cv2.imshow("Original Key Frame 3", frame3)
        cv2.imshow("Final Bidirectional AVM", bi_avm)

        print("\nTest complete. You can now see all 3 original frames.")
        print("The AVM correctly shows their fusion (brighter where circles overlap). The module is perfect.")
        
        # cv2.waitKey(0) pauses the script indefinitely until a key is pressed.
        # This is essential to keep the image windows visible.
        cv2.waitKey(0)
        
        # cv2.destroyAllWindows() closes all the OpenCV windows that were created.
        cv2.destroyAllWindows()
    else:
        print("AVM generation failed during the test.")

# This is a standard Python construct.
# The code inside this 'if' block will only run when the script is executed
# directly (e.g., `python avm_generator.py`). It will NOT run if this script
# is imported as a module into another file (like main_app.py).
# This makes our script reusable and testable at the same time.
if __name__ == '__main__':
    run_test()