from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Screen Dimensions
screen_width, screen_height = pyautogui.size()

# Gesture Variables
last_action_time = 0  # Timer for gesture breaks
gesture_hold_time = 0.8  # Minimum time between gestures

# Helper Functions
def distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks."""
    return ((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)**0.5

def detect_gesture(hand_landmarks, hand_label, img):
    global last_action_time

    # Get relevant landmarks
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Current time for gesture breaks
    current_time = time.time()
    action_text = "None"

    # Cursor Movement (Middle Finger)
    cursor_x = int(middle_tip.x * screen_width)
    cursor_y = int(middle_tip.y * screen_height)
    pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

    # Click Gesture (Index Finger Fully Folded)
    if distance(index_tip, thumb_tip) < 0.09:  # Index tip close to wrist
        if current_time - last_action_time > gesture_hold_time:
            action_text = "Click"
            pyautogui.click()
            last_action_time = current_time

    # Scroll Gesture (Thumb Up or Thumb Down)
    if thumb_tip.y < index_tip.y and thumb_tip.y < pinky_tip.y:  # Thumb up
        if abs(thumb_tip.x - index_tip.x) < 0.2:  # Thumb and index are vertically aligned
            if hand_label == "Right" and current_time - last_action_time > gesture_hold_time:
                action_text = "Scroll Up"
                pyautogui.scroll(60)
                last_action_time = current_time
    elif thumb_tip.y > index_tip.y and thumb_tip.y > pinky_tip.y:  # Thumb down
        if abs(thumb_tip.x - pinky_tip.x) < 0.2:  # Thumb and index are vertically aligned
            if hand_label == "Right" and current_time - last_action_time > gesture_hold_time:
                action_text = "Scroll Down"
                pyautogui.scroll(-60)
                last_action_time = current_time

    # Volume Up Gesture (V Shape with Index and Middle Finger)
    if distance(index_tip, middle_tip) > 0.15:  # Index and middle fingers forming a V
        if hand_label == "Right" and current_time - last_action_time > gesture_hold_time:
            action_text = "Volume Up"
            pyautogui.press("volumeup")
            last_action_time = current_time

    # Volume Down Gesture (Index and Middle Finger Folded)
    if (
        distance(index_tip, middle_tip) < 0.05  # Index and middle tips close together (folded)
    ):
        if hand_label == "Right" and current_time - last_action_time > gesture_hold_time:
            action_text = "Volume Down"
            pyautogui.press("volumedown")
            last_action_time = current_time

    # Zoom Gesture
    if distance(thumb_tip, pinky_tip) < 0.09:  # Thumb and pinky close
        if hand_label == "Right" and current_time - last_action_time > gesture_hold_time:
            action_text = "Zoom In"
            pyautogui.hotkey("ctrl", "+")
            last_action_time = current_time

    # Zoom Gesture
    if distance(thumb_tip, middle_tip) < 0.09:  # Thumb and pinky FAR
        if hand_label == "Right" and current_time - last_action_time > gesture_hold_time:
            action_text = "Zoom OUT"
            pyautogui.hotkey("ctrl", "-")
            last_action_time = current_time

    # Slide Control (Swipe Gestures)
    if distance(thumb_tip, pinky_tip) > 2:  # Wide hand motion
        if hand_label == "Right" and current_time - last_action_time > gesture_hold_time:
            action_text = "Next Slide"
            pyautogui.press("right")
            last_action_time = current_time
        elif hand_label == "Left" and current_time - last_action_time > gesture_hold_time:
            action_text = "Previous Slide"
            pyautogui.press("left")
            last_action_time = current_time

    # Display Action
    cv2.putText(img, f"Action: {action_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def generate_frames():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip and Convert Frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = "Right" if idx == 0 else "Left"  # Assume single hand as right
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                detect_gesture(hand_landmarks, hand_label, frame)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        # Return frame to client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release Resources
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
