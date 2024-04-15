import cv2
import numpy as np
import mediapipe as mp

class AirCanvas:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.track=False
        self.mp_hands = mp.solutions.hands
        self.mp_face_detection = mp.solutions.face_detection 
        self.hands = self.mp_hands.Hands()
        self.face_detection = self.mp_face_detection.FaceDetection()

        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.marker_color = (0, 0, 255)
        self.marker_radius = 10
        self.marker_thickness = 5
        self.marker_position = None
        self.drawing = False
        self.erasing = False
        self.last_position = None

    def draw_marker(self, frame, position):
        if position is not None:
            cv2.circle(frame, position, self.marker_radius, self.marker_color, self.marker_thickness)

    def draw_on_canvas(self, position):
        if self.last_position is not None and position is not None and self.drawing:
            cv2.line(self.canvas, self.last_position, position, self.marker_color, self.marker_thickness)
        self.last_position = position

    def erase_on_canvas(self, position):
        if position is not None:
            cv2.circle(self.canvas, position, self.marker_radius * 2, (0, 0, 0), -1)

    def clear_canvas(self):
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    def detect_face(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'Face {int(detection.score[0] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def run(self):
        cv2.namedWindow("Air Canvas", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Drawing", cv2.WINDOW_NORMAL)

        while True:
            
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands
            
            results_hands = self.hands.process(rgb_frame)

            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    index_finger = hand_landmarks.landmark[8] if len(hand_landmarks.landmark) > 8 else None
                    middle_finger = hand_landmarks.landmark[12] if len(hand_landmarks.landmark) > 12 else None
                    palm = hand_landmarks.landmark[0] if len(hand_landmarks.landmark) > 0 else None

                    if index_finger:
                        # Calculate marker position
                        marker_x = int(index_finger.x * frame.shape[1])
                        marker_y = int(index_finger.y * frame.shape[0])
                        self.marker_position = (marker_x, marker_y)

                        # Determine drawing and erasing states based on finger positions
                        if middle_finger:
                            self.drawing = middle_finger.y >= index_finger.y
                        else:
                            self.drawing = True

                        if palm:
                            self.erasing = palm.y < index_finger.y
                        else:
                            self.erasing=False

                # Erase on canvasq
                if self.erasing:
                    self.erase_on_canvas(self.marker_position)

            # Draw on canvas
            if(self.track==True):
                self.draw_on_canvas(self.marker_position)
            paintWindow = np.zeros((471, 636, 4), dtype=np.uint8)
            paintWindow[:, :, 3] = 0  # Set the alpha channel to 0 (transparent)
            paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0, 255), 2)
            
            # Draw marker
            self.draw_marker(frame, self.marker_position)

            # Overlay canvas on frame
            frame = cv2.addWeighted(frame, 1, self.canvas, 0.5, 0)

            # Show frame
            cv2.imshow("Air Canvas", frame)

            # Show drawing
            cv2.imshow("Drawing", self.canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.imwrite('paint_window.jpg',paintWindow)
                break
            elif key == ord("d"):
                self.clear_canvas()
            elif key==ord("t"):
                if self.marker_color == (255,0,0):
                   self.marker_color=(0,0,255)
                else:
                    self.marker_color=(255,0,0)
                self.track=not self.track
                self.last_position=None

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    air_canvas = AirCanvas()
    air_canvas.run()