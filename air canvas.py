import cv2
import numpy as np
import mediapipe as mp

class AirCanvas:
    def __init__(self):
        # Initialize video capture from the default camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)  # Set width
        self.cap.set(4, 720)   # Set height
        self.track=False #To track drawing or erasing
        # Initialize MediaPipe Hands and Face Detection modules
        self.mp_hands = mp.solutions.hands
        self.mp_face_detection = mp.solutions.face_detection 
        self.hands = self.mp_hands.Hands()
        self.face_detection = self.mp_face_detection.FaceDetection()

        # Initialize canvas for drawing
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.marker_color = (0, 0, 255)  # Marker color (initially red)
        self.marker_radius = 10           # Marker radius
        self.marker_thickness = 5         # Marker thickness
        self.marker_position = None       # Position of the marker
        self.drawing = False              # Flag to indicate drawing state
        self.erasing = False              # Flag to indicate erasing state
        self.last_position = None         # Last position of the marker

    def draw_marker(self, frame, position):
        # Draw marker on the frame
        if position is not None:
            cv2.circle(frame, position, self.marker_radius, self.marker_color, self.marker_thickness)

    def draw_on_canvas(self, position):
        # Draw on canvas
        if self.last_position is not None and position is not None and self.drawing:
            cv2.line(self.canvas, self.last_position, position, self.marker_color, self.marker_thickness)
        self.last_position = position

    def erase_on_canvas(self, position):
        # Erase on canvas
        if position is not None:
            cv2.circle(self.canvas, position, self.marker_radius * 2, (0, 0, 0), -1)

    def clear_canvas(self):
        # Clear canvas
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    def detect_face(self, frame):
        # Detect face in the frame
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
        # Create windows to display the Air Canvas and Drawing
        cv2.namedWindow("Air Canvas", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Drawing", cv2.WINDOW_NORMAL)

        while True:
            # Capture frame from the camera
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

                # Erase on canvas
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

            # Check for key press events
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.imwrite('paint_window.jpg',paintWindow)  # Save the drawing window as an image
                break
            elif key == ord("d"):
                self.clear_canvas()  # Clear the canvas
            elif key==ord("t"):
                if self.marker_color == (255,0,0):  # Toggle marker color between red and blue
                   self.marker_color=(0,0,255)
                else:
                    self.marker_color=(255,0,0)
                self.track=not self.track  # Toggle drawing tracking state
                self.last_position=None   # Reset last position

        # Release the camera and close windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create an instance of the AirCanvas class and run the application
    air_canvas = AirCanvas()
    air_canvas.run()
