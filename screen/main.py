import cv2
import mediapipe as mp
import pyautogui

# Initialize hand tracking module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize PyAutoGUI
pyautogui.FAILSAFE = False

# Function to check thumbs up and two fingers up gestures
def check_hand_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Calculate the y-coordinates of thumb tip, index finger tip, middle finger tip, and wrist
    thumb_y = thumb_tip.y
    index_y = index_tip.y
    middle_y = middle_tip.y
    wrist_y = wrist.y

    # Adjust these values according to your hand size and position
    thumb_threshold_up = 0.8
    fingers_up_threshold = 0.8
    wrist_threshold = 0.5

    # Check thumbs up gesture
    if thumb_y < index_y and thumb_y < thumb_threshold_up:
        return "thumbs_up"

    # Check two fingers up gesture
    elif index_y < wrist_y and middle_y < wrist_y and index_y < fingers_up_threshold and middle_y < fingers_up_threshold:
        return "two_fingers_up"


    return "no_gesture"

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        results = hands.process(rgb_frame)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check hand gesture
                gesture = check_hand_gesture(hand_landmarks)


                # Perform scrolling based on hand gesture
                if gesture == "thumbs_up":
                    pyautogui.scroll(50)  # Increased scrolling speed for thumbs up
                elif gesture == "two_fingers_up":
                    pyautogui.scroll(-50)  # Increased scrolling speed for two fingers up
                else:
                    pyautogui.scroll(0)  # No scrolling for other gestures

        # Display the frame
        cv2.imshow('Hand Tracking', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
