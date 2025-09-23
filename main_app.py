import sys
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from avm_generator import generate_avm

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

STYLESHEET = """
QWidget#MainWindow { background-color: #f0f0f0; }
QLabel#TitleLabel { font-family: Arial; font-size: 16px; font-weight: bold; color: #555; }
QWidget#ContentPane { background-color: #ffffff; border-radius: 15px; }
QLabel#DisplayLabel { background-color: #000000; color: #ffffff; font-size: 20px; border-radius: 10px; }
QTextEdit#TranslationText { background-color: #ffffff; border: none; color: #333; font-size: 16px; }
QPushButton { background-color: #d0d0d0; border: none; padding: 15px; font-size: 16px; font-weight: bold; border-radius: 10px; }
QPushButton:hover { background-color: #c0c0c0; }
"""

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    # --- NEW: A separate signal for the AVM image ---
    new_avm_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # --- NEW: Key posture logic variables moved here ---
        self.key_frames_buffer = deque(maxlen=16)
        self.wrist_velocity_threshold = 0.01
        self.previous_wrist_coords = None
        self.is_hand_paused = False

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag and cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Make a clean copy for the buffer before any drawings
            original_image_for_buffer = image.copy()

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            image.flags.writeable = True

            # --- NEW: Full key posture and AVM trigger logic ---
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                current_wrist_coords = np.array([wrist.x, wrist.y])

                if self.previous_wrist_coords is not None:
                    velocity = np.linalg.norm(current_wrist_coords - self.previous_wrist_coords)
                    if velocity < self.wrist_velocity_threshold:
                        if not self.is_hand_paused:
                            self.key_frames_buffer.append(original_image_for_buffer)
                            print(f"Key posture captured. Buffer size: {len(self.key_frames_buffer)}")
                            self.is_hand_paused = True
                    else:
                        self.is_hand_paused = False
                self.previous_wrist_coords = current_wrist_coords
            else: # No hand detected
                if len(self.key_frames_buffer) > 0:
                    print("Hand removed. Generating AVM...")
                    _, _, bidirectional_avm = generate_avm(list(self.key_frames_buffer))
                    if bidirectional_avm is not None:
                        # Emit the AVM image on the new signal
                        self.new_avm_signal.emit(bidirectional_avm)
                    self.key_frames_buffer.clear()
                    print("Buffer cleared.")
                self.previous_wrist_coords = None
                self.is_hand_paused = False
            
            # Emit the live video frame for display
            self.change_pixmap_signal.emit(cv2.flip(image, 1))
        
        cap.release()
        self.hands.close()
        print("Video thread finished.")

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.setup_ui()

    def setup_ui(self):
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
        self.video_feed_label.setObjectName("DisplayLabel") # Renamed for consistency
        self.video_feed_label.setAlignment(Qt.AlignCenter)
        left_pane_layout.addWidget(left_title)
        left_pane_layout.addWidget(self.video_feed_label, 1)

        right_pane = QWidget()
        right_pane.setObjectName("ContentPane")
        right_pane_layout = QVBoxLayout(right_pane)
        right_title = QLabel("DIAGNOSTICS & TRANSLATION")
        right_title.setObjectName("TitleLabel")
        right_title.setAlignment(Qt.AlignCenter)
        
        # --- NEW: Separate label for the AVM ---
        self.avm_display_label = QLabel("AVM will appear here")
        self.avm_display_label.setObjectName("DisplayLabel")
        self.avm_display_label.setAlignment(Qt.AlignCenter)

        self.translation_text_area = QTextEdit("Translation will appear here...")
        self.translation_text_area.setObjectName("TranslationText")
        self.translation_text_area.setReadOnly(True)
        
        right_pane_layout.addWidget(right_title)
        right_pane_layout.addWidget(self.avm_display_label, 1) # Give AVM view equal space
        right_pane_layout.addWidget(self.translation_text_area, 1) # Give text area equal space

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
        
        self.start_button.clicked.connect(self.start_video_feed)
        self.stop_button.clicked.connect(self.stop_video_feed)

    def start_video_feed(self):
        print("START button clicked!")
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        # --- NEW: Connect the new AVM signal to its own slot ---
        self.thread.new_avm_signal.connect(self.update_avm_image)
        self.thread.start()
        self.translation_text_area.setText("Video feed started. Perform a sign.")

    def stop_video_feed(self):
        print("STOP button clicked!")
        if self.thread:
            self.thread.stop()
        self.video_feed_label.setText("Video feed stopped.")
        # --- NEW: Clear the AVM label on stop ---
        self.avm_display_label.setText("AVM will appear here")
        self.avm_display_label.setStyleSheet("background-color: #000; color: #fff;") # Reset style
        self.translation_text_area.clear()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img, self.video_feed_label)
        self.video_feed_label.setPixmap(qt_img)
    
    # --- NEW: A dedicated slot to handle updating the AVM image ---
    @pyqtSlot(np.ndarray)
    def update_avm_image(self, avm_img):
        print("Updating AVM display.")
        qt_img = self.convert_cv_qt(avm_img, self.avm_display_label)
        self.avm_display_label.setPixmap(qt_img)
        self.translation_text_area.setText("AVM Generated. Ready for prediction...")

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