import cv2
import numpy as np
from datetime import datetime

class WatermarkConfig:
    def __init__(self):
        self.steps = [
            "Step 1/4: Front View",
            "Step 2/4: Back View",
            "Step 3/4: Brand Packaging",
            "Step 4/4: Accessories"
        ]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_color = (255, 255, 255)
        self.bg_color = (0, 0, 0)
        self.alpha = 0.6
        self.thickness = 2
        self.padding = 20


class WatermarkSystem:
    def __init__(self, config):
        self.config = config
        self.current_step = 0
        self.order_id = "ORD123456"
        self.station_code = "NZM80"

    def next_step(self):
        self.current_step = (self.current_step + 1) % len(self.config.steps)

    def previous_step(self):
        self.current_step = (self.current_step - 1) % len(self.config.steps)

    def draw_overlay(self, frame):
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Top overlay bar
        cv2.rectangle(overlay, (0, 0), (width, 100), self.config.bg_color, -1)

        # Bottom overlay bar
        cv2.rectangle(overlay, (0, height - 60), (width, height), self.config.bg_color, -1)

        # Blend overlay
        frame = cv2.addWeighted(overlay, self.config.alpha, frame, 1 - self.config.alpha, 0)

        return frame

    def add_text(self, frame):
        step_text = self.config.steps[self.current_step]

        # Step instruction
        cv2.putText(
            frame,
            step_text,
            (20, 50),
            self.config.font,
            self.config.font_scale,
            self.config.font_color,
            self.config.thickness
        )

        # Metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = f"{self.order_id} | {self.station_code} | {timestamp}"

        cv2.putText(
            frame,
            metadata,
            (20, frame.shape[0] - 20),
            self.config.font,
            0.6,
            self.config.font_color,
            1
        )

        return frame

    def process_frame(self, frame):
        frame = self.draw_overlay(frame)
        frame = self.add_text(frame)
        return frame


def run_watermark_system():
    config = WatermarkConfig()
    system = WatermarkSystem(config)

    cap = cv2.VideoCapture(0)

    print("Press 'n' for next step, 'p' for previous step, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = system.process_frame(frame)

        cv2.imshow("OBD Watermark System", frame)

        key = cv2.waitKey(1)

        if key == ord('n'):
            system.next_step()
        elif key == ord('p'):
            system.previous_step()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_watermark_system()
