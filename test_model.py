# ==============================================================================
# test_model.py (v2 with AVM Visualization)
#
# Description:
# This version now displays the generated AVM image, allowing us to visually
# inspect the input that the AMN stream of our model is receiving.
# ==============================================================================

import os
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception, MobileNetV2
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from avm_generator import generate_avm
from sign_mapper import LSA64_MAP

# (Helper functions remain the same)
def frames_downsample(arFrames:np.array, nFramesTarget:int) -> np.array:
    nFramesExisting = arFrames.shape[0]
    if nFramesExisting == nFramesTarget: return arFrames
    indices = np.linspace(0, nFramesExisting - 1, nFramesTarget, dtype=int)
    return arFrames[indices, ...]

def srn_build(nFramesNorm, nFeatureLength, nClasses):
    input_frames = Input(shape=(nFramesNorm, nFeatureLength), name='input_dmn_features')
    x1 = LSTM(2048, return_sequences=True, dropout=0.5)(input_frames)
    x1 = LSTM(2048, return_sequences=False, dropout=0.5)(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    img_size = 224
    input_img = Input(shape=(img_size, img_size, 3), name='input_amn_image')
    x2_processed = mobilenet_preprocess(input_img)
    base_cnn = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    base_cnn.trainable = False
    x2 = base_cnn(x2_processed, training=False)
    x2 = GlobalAveragePooling2D()(x2)
    x = concatenate([x1, x2])
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    fc = Dense(nClasses, activation="softmax")(x)
    model = Model(inputs=[input_frames, input_img], outputs=fc)
    return model

# In test_model.py, replace the main() function:

def main():
    # --- 1. CONFIGURATION ---
    TEST_VIDEO_PATH = "005_004_002.mp4"
    EXPECTED_SIGN = "Bright"

    print(f"--- Starting Sanity Check for video: {TEST_VIDEO_PATH} ---")
    print(f"Expected Prediction: {EXPECTED_SIGN}")

    # (Model loading is the same)
    print("\nLoading models and class names...")
    try:
        with open('class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("FATAL ERROR: class_names.txt not found.")
        return
    num_classes = len(class_names)
    srn_model = srn_build(18, 2048, num_classes)
    srn_model.load_weights('srn_model.h5')
    base_xception = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    x = base_xception.output
    output = GlobalAveragePooling2D()(x)
    feature_extractor = Model(inputs=base_xception.input, outputs=output)
    print("Models loaded successfully.")

    # (Video processing is the same)
    print("\nProcessing video file...")
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    frame_buffer = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_buffer.append(frame)
    cap.release()
    print(f"Extracted {len(frame_buffer)} frames from the video.")
    if len(frame_buffer) < 5:
        print("Error: Not enough frames in the video to process.")
        return

    # (AVM generation is the same)
    _, _, avm_image = generate_avm(frame_buffer)
    if avm_image is None:
        print("Error: Failed to generate AVM image.")
        return
    print("AVM image generated successfully.")

    # (Prediction pipeline is the same)
    print("\nRunning prediction pipeline...")
    dmn_frames = frames_downsample(np.array(frame_buffer), 18)
    dmn_frames_resized = np.array([cv2.resize(f, (299, 299)) for f in dmn_frames])
    dmn_frames_preprocessed = xception_preprocess(dmn_frames_resized)
    dmn_features = feature_extractor.predict(dmn_frames_preprocessed, verbose=0)
    dmn_input = np.expand_dims(dmn_features, axis=0)
    amn_image_resized = cv2.resize(avm_image, (224, 224))
    amn_image_preprocessed = mobilenet_preprocess(amn_image_resized)
    amn_input = np.expand_dims(amn_image_preprocessed, axis=0)
    prediction = srn_model.predict([dmn_input, amn_input], verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_class_id = class_names[predicted_index]
    sign_id = predicted_class_id.split('_')[0]
    predicted_word = LSA64_MAP.get(sign_id, "UNKNOWN SIGN")
    confidence = prediction[0][predicted_index]

    # (Show results is the same)
    print("\n--- PREDICTION RESULTS ---")
    print(f"Predicted Word: {predicted_word.upper()}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Expected Word:  {EXPECTED_SIGN.upper()}")
    if predicted_word.upper() == EXPECTED_SIGN.upper():
        print("\nSUCCESS: The model correctly identified the sign from the dataset!")
    else:
        print("\nFAILURE: The model did not identify the sign correctly. There may be an issue.")

    # --- NEW: Resize the AVM image before displaying ---
    # Let's resize it to a fixed height of 480 pixels, keeping the aspect ratio.
    h, w = avm_image.shape[:2]
    scale_factor = 480 / h
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    avm_display = cv2.resize(avm_image, (new_w, new_h))
    
    cv2.imshow("Generated AVM for " + TEST_VIDEO_PATH, avm_display)
    print("\nDisplaying AVM image. Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()