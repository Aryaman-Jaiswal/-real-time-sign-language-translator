import cv2
import numpy as np
import random

def random_rotate(image, angle_range=(-10, 10)):
    """
    Rotates an image by a random angle within the specified range.
    The image is scaled to fit the new dimensions without being cropped.
    """
    angle = random.uniform(angle_range[0], angle_range[1])
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get the rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rot_mat, (w, h))
    return rotated_image

def random_brightness_contrast(image, brightness_range=(-50, 50), contrast_range=(0.7, 1.3)):
    """
    Adjusts the brightness and contrast of an image by random values.
    """
    brightness = random.randint(brightness_range[0], brightness_range[1])
    contrast = random.uniform(contrast_range[0], contrast_range[1])
    
    # Apply the contrast and brightness adjustment
    # The formula is: new_image = alpha * original_image + beta
    # where alpha is contrast and beta is brightness
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted_image

def random_translate(image, translate_range=(-20, 20)):
    """
    Shifts an image horizontally and vertically by a random amount.
    """
    tx = random.randint(translate_range[0], translate_range[1])
    ty = random.randint(translate_range[0], translate_range[1])
    h, w = image.shape[:2]
    
    # Create the translation matrix
    trans_mat = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Perform the translation
    translated_image = cv2.warpAffine(image, trans_mat, (w, h))
    return translated_image

def run_augmentation_test():
    """
    Loads a sample image, applies augmentations, and displays them
    in a single, combined window for easy viewing and screenshots.
    """
    try:
        # Load a sample image.
        original_image = cv2.imread("test_hand.png") # Changed to .png
        if original_image is None:
            raise FileNotFoundError("Make sure you have a 'test_hand.png' image in your folder.")

    except FileNotFoundError as e:
        print(e)
        print("Creating a dummy image for testing.")
        original_image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(original_image, 'ORIGINAL', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # --- NEW: Resize to a manageable size first ---
    # This ensures the final combined window isn't too large.
    display_size = (320, 240) # Width, Height
    original_image = cv2.resize(original_image, display_size)

    # Apply the augmentations to the resized image
    rotated = random_rotate(original_image)
    bright_contrast = random_brightness_contrast(original_image)
    translated = random_translate(original_image)
    
    # --- NEW: Add labels to each image for clarity ---
    cv2.putText(original_image, 'Original', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(rotated, 'Rotated', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(bright_contrast, 'Bright/Contrast', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(translated, 'Translated', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # --- NEW: Combine the images into a 2x2 grid ---
    # Create the top row by stacking images horizontally
    top_row = np.hstack([original_image, rotated])
    # Create the bottom row
    bottom_row = np.hstack([bright_contrast, translated])
    
    # Combine the top and bottom rows vertically
    combined_image = np.vstack([top_row, bottom_row])
    
    # Display the single, combined image
    cv2.imshow("Data Augmentation Showcase", combined_image)
    
    print("Augmentation test running. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_augmentation_test()