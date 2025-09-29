# ==============================================================================
# data_augmentation.py
#
# Author: Aryaman Jaiswal 
#
# Description:
# This module provides a set of functions for performing data augmentation on
# video frames. Data augmentation is a critical technique in machine learning
# for artificially expanding a dataset. By creating modified versions of the
# training images, we expose the model to a wider variety of conditions, which
# helps it learn more robust features and improves its ability to generalize to
# new, unseen data (like different signers or lighting conditions).
#
# The script also includes a self-contained test function, run_augmentation_test(),
# to visually demonstrate the effects of each augmentation technique.
# ==============================================================================

import cv2
import numpy as np
import random

# --- Augmentation Functions ---

def random_rotate(image, angle_range=(-10, 10)):
    """
    Rotates an image by a random angle within a specified range.
    This simulates variations in camera tilt.

    Args:
        image (np.ndarray): The input image to rotate.
        angle_range (tuple[int, int]): A tuple specifying the min and max
                                       rotation angle in degrees.

    Returns:
        np.ndarray: The rotated image.
    """
    # Select a random angle from the specified range.
    angle = random.uniform(angle_range[0], angle_range[1])
    
    # Get the image dimensions (height, width) and calculate the center.
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate the 2D rotation matrix using OpenCV.
    # Arguments are: center of rotation, angle, and scale factor (1.0 means no scaling).
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply the rotation to the image. An affine transformation shifts, scales,
    # and rotates the image according to the matrix. The output dimensions (w, h)
    # are kept the same, so parts of the rotated image may go out of bounds.
    rotated_image = cv2.warpAffine(image, rot_mat, (w, h))
    return rotated_image

def random_brightness_contrast(image, brightness_range=(-50, 50), contrast_range=(0.7, 1.3)):
    """
    Randomly adjusts the brightness and contrast of an image.
    This simulates different real-world lighting conditions.

    Args:
        image (np.ndarray): The input image.
        brightness_range (tuple[int, int]): Range for the random brightness value.
                                            Negative values darken, positive brighten.
        contrast_range (tuple[float, float]): Range for the random contrast value.
                                              Values < 1.0 decrease contrast, > 1.0 increase it.

    Returns:
        np.ndarray: The image with adjusted brightness and contrast.
    """
    # Choose random values for brightness and contrast from their respective ranges.
    brightness = random.randint(brightness_range[0], brightness_range[1])
    contrast = random.uniform(contrast_range[0], contrast_range[1])
    
    # OpenCV's adjustment formula is: new_pixel = alpha * old_pixel + beta
    # where alpha is the contrast and beta is the brightness.
    # cv2.convertScaleAbs applies this transformation and ensures the final
    # pixel values are valid (i.e., clipped to the 0-255 range).
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted_image

def random_translate(image, translate_range=(-20, 20)):
    """
    Randomly shifts (translates) an image horizontally and vertically.
    This simulates the user not being perfectly centered in the camera frame.

    Args:
        image (np.ndarray): The input image.
        translate_range (tuple[int, int]): The min and max pixel shift for both
                                           the x and y directions.

    Returns:
        np.ndarray: The translated image.
    """
    # Choose random pixel shifts for the x (tx) and y (ty) directions.
    tx = random.randint(translate_range[0], translate_range[1])
    ty = random.randint(translate_range[0], translate_range[1])
    h, w = image.shape[:2]
    
    # Create the 2x3 translation matrix.
    # [1, 0, tx] means: no x-scaling, no y-shear, shift by tx pixels.
    # [0, 1, ty] means: no x-shear, no y-scaling, shift by ty pixels.
    trans_mat = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply the translation using a warp affine transformation.
    translated_image = cv2.warpAffine(image, trans_mat, (w, h))
    return translated_image

# --- Test and Visualization Function ---

def run_augmentation_test():
    """
    A test harness to load a sample image and visually demonstrate the
    effects of the augmentation functions in a single, combined window.
    """
    try:
        # Attempt to load a local image file for the demonstration.
        original_image = cv2.imread("test_hand.png")
        # cv2.imread returns None if the file is not found or cannot be read.
        if original_image is None:
            raise FileNotFoundError("Make sure you have a 'test_hand.png' image in your folder.")
    except FileNotFoundError as e:
        # If the image file doesn't exist, create a dummy image so the
        # script can still run and demonstrate the functionality.
        print(e)
        print("Creating a dummy image for testing.")
        original_image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(original_image, 'ORIGINAL', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Resize the source image to a consistent, manageable size for display.
    # This prevents the final combined window from being excessively large.
    display_size = (320, 240) # (Width, Height)
    original_image = cv2.resize(original_image, display_size)

    # --- Apply Each Augmentation ---
    rotated = random_rotate(original_image)
    bright_contrast = random_brightness_contrast(original_image)
    translated = random_translate(original_image)
    
    # --- Prepare Images for Display ---
    # Draw text labels directly onto each image to identify them in the final view.
    cv2.putText(original_image, 'Original', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(rotated, 'Rotated', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(bright_contrast, 'Bright/Contrast', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(translated, 'Translated', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # --- Combine Images into a 2x2 Grid ---
    # np.hstack() stacks arrays horizontally (side-by-side).
    top_row = np.hstack([original_image, rotated])
    bottom_row = np.hstack([bright_contrast, translated])
    
    # np.vstack() stacks arrays vertically (one on top of the other).
    combined_image = np.vstack([top_row, bottom_row])
    
    # --- Display the Final Showcase Window ---
    cv2.imshow("Data Augmentation Showcase", combined_image)
    
    print("Augmentation test running. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# This standard Python construct ensures the test code only runs when the
# script is executed directly.
if __name__ == "__main__":
    run_augmentation_test()