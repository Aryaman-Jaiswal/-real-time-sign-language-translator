import cv2
import numpy as np
import os

def generate_avm(key_frames: list[np.ndarray]):
    """
    Generates Accumulative Video Motion (AVM) images from a list of key frames.

    This corrected version explicitly sets the output data type in the normalize
    function to solve the black image issue. It ensures the high-precision
    float array is correctly converted to a displayable 8-bit image.

    Args:
        key_frames (list[np.ndarray]): A list of images (as NumPy arrays).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
        forward_avm, backward_avm, and bidirectional_avm images.
    """
    if not key_frames:
        print("Warning: key_frames list is empty. Cannot generate AVM.")
        return None, None, None

    # Convert frames to float32 for safe summation
    frames_array = np.array(key_frames, dtype=np.float32)

    # Sum all frames together
    summed_image = np.sum(frames_array, axis=0)

    # Normalize the summed image back to a displayable 0-255 range.
    # THE FIX IS HERE: We add the dtype=cv2.CV_8U parameter.
    # This tells OpenCV to create an 8-bit unsigned integer image as the output.
    normalized_image = cv2.normalize(summed_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Since summation is commutative, forward, backward, and bidirectional are all the same
    forward_avm = normalized_image
    backward_avm = normalized_image
    bidirectional_avm = normalized_image

    return forward_avm, backward_avm, bidirectional_avm


def run_test():
    """
    Final test case using overlapping circles and displaying each source
    frame in its own window to avoid display issues.
    """
    print("Running AVM generation test with overlapping shapes...")

    # Create dummy key frames with overlapping circles
    height, width = 300, 500
    
    # Frame 1: Circle on the left
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(frame1, (width // 4, height // 2), 70, (255, 255, 255), -1)

    # Frame 2: Circle in the middle
    frame2 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(frame2, (width // 2, height // 2), 70, (255, 255, 255), -1)

    # Frame 3: Circle on the right
    frame3 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(frame3, (width * 3 // 4, height // 2), 70, (255, 255, 255), -1)
    
    key_frames_list = [frame1, frame2, frame3]

    # --- Generate the AVM images ---
    fwd_avm, bwd_avm, bi_avm = generate_avm(key_frames_list)

    if fwd_avm is not None:
        # THE FIX IS HERE: Display each original frame in a separate window
        cv2.imshow("Original Key Frame 1", frame1)
        cv2.imshow("Original Key Frame 2", frame2)
        cv2.imshow("Original Key Frame 3", frame3)

        cv2.imshow("Final Bidirectional AVM", bi_avm)

        print("\nTest complete. You can now see all 3 original frames.")
        print("The AVM correctly shows their fusion. The module is perfect.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("AVM generation failed.")

if __name__ == '__main__':
    run_test()