import cv2
import numpy as np

class DetectionConfig:
    def __init__(self):
        self.blur_threshold = 100
        self.face_scale_factor = 1.3
        self.face_neighbors = 5
        self.font = cv2.FONT_HERSHEY_SIMPLEX


class QualityDetectionSystem:
    def __init__(self, config):
        self.config = config
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def compute_blur(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = variance < self.config.blur_threshold
        return is_blurry, variance

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            self.config.face_scale_factor,
            self.config.face_neighbors
        )
        return faces

    def draw_warnings(self, frame, is_blurry, faces, blur_score):
        y_offset = 40

        if is_blurry:
            cv2.putText(
                frame,
                "Blur Detected! Please Retake",
                (20, y_offset),
                self.config.font,
                0.8,
                (0, 0, 255),
                2
            )
            y_offset += 40

        if len(faces) > 0:
            cv2.putText(
                frame,
                "Face Detected! Avoid Customer",
                (20, y_offset),
                self.config.font,
                0.8,
                (0, 0, 255),
                2
            )

        # Draw face boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Debug blur score
        cv2.putText(
            frame,
            f"Blur Score: {blur_score:.2f}",
            (20, frame.shape[0] - 20),
            self.config.font,
            0.6,
            (255, 255, 255),
            1
        )

        return frame

    def process_frame(self, frame):
        is_blurry, blur_score = self.compute_blur(frame)
        faces = self.detect_faces(frame)
        frame = self.draw_warnings(frame, is_blurry, faces, blur_score)
        return frame


def run_detection_system():
    config = DetectionConfig()
    system = QualityDetectionSystem(config)

    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = system.process_frame(frame)

        cv2.imshow("OBD Quality Detection System", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection_system()
