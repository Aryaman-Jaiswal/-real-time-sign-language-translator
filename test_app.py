# ==============================================================================
#           FINAL APPLICATION: main_app.py (Timed Recording Version)
#
# Author: Aryaman Jaiswal (with guidance from an AI assistant)
#
# Description:
# This final version implements a new, user-controlled recording workflow.
# Instead of continuous motion detection, the user clicks START to begin a
# fixed 3-second recording session. Frames are captured at a set interval,
# and prediction occurs automatically at the end.
# ==============================================================================

import sys
import os
import cv2
import numpy as np
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception, MobileNetV2
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer

from avm_generator import generate_avm
from sign_mapper import LSA64_MAP

# (Helper functions srn_build and frames_downsample remain the same)
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

STYLESHEET = """
QWidget#MainWindow { background-color: #f0f0f0; }
QLabel#TitleLabel { font-family: Arial; font-size: 16px; font-weight: bold; color: #555; }
QWidget#ContentPane { background-color: #ffffff; border-radius: 15px; }
QLabel#DisplayLabel { background-color: #000000; color: #ffffff; font-size: 20px; border-radius: 10px; }
QTextEdit#TranslationText { background-color: #ffffff; border: none; color: #008000; font-size: 24px; font-weight: bold; }
QPushButton { background-color: #d0d0d0; border: none; padding: 15px; font-size: 16px; font-weight: bold; border-radius: 10px; }
QPushButton:hover { background-color: #c0c0c0; }
QPushButton:disabled { background-color: #f0f0f0; color: #a0a0a0; }
"""

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    new_avm_signal = pyqtSignal(np.ndarray)
    new_prediction_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.latest_frame = None
        self.frame_buffer = []
        
        # --- NEW: Timers for the recording logic ---
        self.capture_timer = QTimer()
        self.capture_timer.setInterval(33) # Capture a frame every 50ms (0.5s)
        self.capture_timer.timeout.connect(self.capture_frame)
        
        self.recording_timer = QTimer()
        self.recording_timer.setSingleShot(True) # This timer only runs once
        self.recording_timer.setInterval(3000) # Stop recording after 3000ms (3s)
        self.recording_timer.timeout.connect(self.stop_recording)

        # (Model loading remains the same)
        print("Loading models and class names, please wait...")
        try:
            with open('class_names.txt', 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            self.class_names = []
        self.sign_map = LSA64_MAP
        num_classes = len(self.class_names) if self.class_names else 640
        self.srn_model = srn_build(18, 2048, num_classes)
        self.srn_model.load_weights('srn_model.h5')
        base_xception = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        x = base_xception.output
        output = GlobalAveragePooling2D()(x)
        self.feature_extractor = Model(inputs=base_xception.input, outputs=output)
        print("Models and class names loaded successfully.")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag and cap.isOpened():
            success, image = cap.read()
            if not success: continue
            
            # Store the latest frame for the capture timer to grab
            self.latest_frame = image.copy()

            # MediaPipe processing for visual feedback
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            self.change_pixmap_signal.emit(cv2.flip(image, 1))
        cap.release()
        self.hands.close()

    def start_recording(self):
        """Called by the main window to start the timed capture."""
        self.frame_buffer.clear()
        self.capture_timer.start()
        self.recording_timer.start()
        print("Recording started for 3 seconds...")

    def capture_frame(self):
        """Called every 0.5s by the capture_timer."""
        if self.latest_frame is not None:
            self.frame_buffer.append(self.latest_frame)
            print(f"Frame captured. Total frames: {len(self.frame_buffer)}")

    # In VideoThread class, find this function:
    def stop_recording(self):
        """Called after 3s by the recording_timer."""
        self.capture_timer.stop()
        print("Recording stopped. Processing frames...")
        
        # --- TEMPORARY CHANGE FOR DEBUGGING ---
        print("!!! OVERRIDE: Loading frames from test video file instead of webcam !!!")
        
        # Load the frames from the known-good video file
        cap = cv2.VideoCapture("005_004_002.mp4")
        debug_frame_buffer = []
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            debug_frame_buffer.append(frame)
        cap.release()
        
        # Use this known-good buffer instead of the live one
        self.frame_buffer = debug_frame_buffer
        # --- END OF TEMPORARY CHANGE ---

        if len(self.frame_buffer) > 0:
            _, _, bidirectional_avm = generate_avm(self.frame_buffer)
            if bidirectional_avm is not None:
                self.new_avm_signal.emit(bidirectional_avm)
                self.predict_sign(self.frame_buffer, bidirectional_avm)
        else:
            self.new_prediction_signal.emit("Capture failed.")


    def predict_sign(self, frame_buffer, avm_image):
        # (Prediction logic is the same)
        dmn_frames = frames_downsample(np.array(frame_buffer), 18)
        dmn_frames_resized = np.array([cv2.resize(f, (299, 299)) for f in dmn_frames])
        dmn_frames_preprocessed = xception_preprocess(dmn_frames_resized)
        dmn_features = self.feature_extractor.predict(dmn_frames_preprocessed, verbose=0)
        dmn_input = np.expand_dims(dmn_features, axis=0)
        amn_image_resized = cv2.resize(avm_image, (224, 224))
        amn_image_preprocessed = mobilenet_preprocess(amn_image_resized)
        amn_input = np.expand_dims(amn_image_preprocessed, axis=0)
        prediction = self.srn_model.predict([dmn_input, amn_input], verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_class_id = self.class_names[predicted_index]
        sign_id = predicted_class_id.split('_')[0]
        predicted_word = self.sign_map.get(sign_id, "UNKNOWN SIGN")
        confidence = prediction[0][predicted_index]
        final_text_to_display = f"{predicted_word.upper()}\n(Confidence: {confidence:.2f})"
        print(f"Prediction: {final_text_to_display}")
        self.new_prediction_signal.emit(final_text_to_display)


         # Create a directory to save our debug files if it doesn't exist
        debug_dir = "debug_output"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            
        # 1. Save the AVM image the model is seeing
        cv2.imwrite(os.path.join(debug_dir, "LAST_AVM_INPUT.jpg"), avm_image)
        
        # 2. Save the 18 frames the DMN is seeing
        dmn_frames_for_debug = frames_downsample(np.array(frame_buffer), 18)
        for i, frame in enumerate(dmn_frames_for_debug):
            cv2.imwrite(os.path.join(debug_dir, f"LAST_DMN_FRAME_{i:02d}.jpg"), frame)
            
        print("!!! DEBUG: Saved AVM and DMN frames to 'debug_output' folder. !!!")

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QWidget):
    # (The MainWindow class is now simpler)
    def __init__(self):
        super().__init__()
        self.thread = None
        self.is_recording = False
        self.setup_ui()

    def setup_ui(self):
        # (UI setup is the same)
        self.setObjectName("MainWindow")
        self.setWindowTitle("Real-Time Sign Language Translator")
        self.setGeometry(100, 100, 1200, 700)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        top_layout = QHBoxLayout()
        left_pane = QWidget()
        left_pane.setObjectName("ContentPane")
        left_pane_layout = QVBoxLayout(left_pane)
        left_title = QLabel("LIVE VIDEO FEED")
        left_title.setObjectName("TitleLabel")
        left_title.setAlignment(Qt.AlignCenter)
        self.video_feed_label = QLabel("Press START to begin")
        self.video_feed_label.setObjectName("DisplayLabel")
        self.video_feed_label.setAlignment(Qt.AlignCenter)
        left_pane_layout.addWidget(left_title)
        left_pane_layout.addWidget(self.video_feed_label, 1)
        right_pane = QWidget()
        right_pane.setObjectName("ContentPane")
        right_pane_layout = QVBoxLayout(right_pane)
        right_title = QLabel("DIAGNOSTICS & TRANSLATION")
        right_title.setObjectName("TitleLabel")
        right_title.setAlignment(Qt.AlignCenter)
        self.avm_display_label = QLabel("AVM will appear here")
        self.avm_display_label.setObjectName("DisplayLabel")
        self.avm_display_label.setAlignment(Qt.AlignCenter)
        self.translation_text_area = QTextEdit("Translation will appear here...")
        self.translation_text_area.setObjectName("TranslationText")
        self.translation_text_area.setReadOnly(True)
        right_pane_layout.addWidget(right_title)
        right_pane_layout.addWidget(self.avm_display_label, 1)
        right_pane_layout.addWidget(self.translation_text_area, 1)
        top_layout.addWidget(left_pane, 2)
        top_layout.addWidget(right_pane, 1)
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        self.start_button = QPushButton("START")
        self.stop_button = QPushButton("STOP")
        bottom_layout.addWidget(self.start_button)
        bottom_layout.addWidget(self.stop_button)
        bottom_layout.addStretch()
        main_layout.addLayout(top_layout, 1)
        main_layout.addLayout(bottom_layout)
        self.start_button.clicked.connect(self.toggle_recording)
        self.stop_button.clicked.connect(self.stop_video_feed)
        self.stop_button.setEnabled(False)

    def toggle_recording(self):
        """Starts the video feed if it's not running, or starts the recording if it is."""
        if not self.thread or not self.thread.isRunning():
            # This is the first click: start the video feed
            self.thread = VideoThread()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.new_avm_signal.connect(self.update_avm_image)
            self.thread.new_prediction_signal.connect(self.update_prediction_text)
            self.thread.start()
            self.start_button.setText("RECORD SIGN")
            self.stop_button.setEnabled(True)
            self.translation_text_area.setText("Ready. Press 'RECORD SIGN' to capture.")
        else:
            # This is the second click: start the 3-second recording
            self.thread.start_recording()
            self.translation_text_area.setText("RECORDING...")
            self.start_button.setEnabled(False) # Disable button during recording

    def stop_video_feed(self):
        if self.thread:
            self.thread.stop()
        self.video_feed_label.setText("Video feed stopped.")
        self.avm_display_label.setText("AVM will appear here")
        self.avm_display_label.setStyleSheet("background-color: #000; color: #fff;")
        self.translation_text_area.clear()
        self.start_button.setText("START")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img, self.video_feed_label)
        self.video_feed_label.setPixmap(qt_img)
    
    @pyqtSlot(np.ndarray)
    def update_avm_image(self, avm_img):
        qt_img = self.convert_cv_qt(avm_img, self.avm_display_label)
        self.avm_display_label.setPixmap(qt_img)

    @pyqtSlot(str)
    def update_prediction_text(self, word):
        self.translation_text_area.setText(word)
        self.start_button.setEnabled(True) # Re-enable button after prediction

    def convert_cv_qt(self, cv_img, target_label):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = QPixmap.fromImage(convert_to_Qt_format)
        return p.scaled(target_label.width(), target_label.height(), Qt.KeepAspectRatio)

    def closeEvent(self, event):
        self.stop_video_feed()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())