import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Start video capture
cap = cv2.VideoCapture(0)

# Prevents undefined variable error
left_click_active = False  
right_click_active = False  
last_click_time = 0
click_cooldown = 0.5
right_click_cooldown = 0.5
double_click_threshold = 0.3  # seconds
previous_y = None  # Initialize previous y-position
previous_zoom_distance = None  # Initialize previous zoom distance

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get current time
    current_time = time.time()

    # Flip frame horizontally and convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand detection
    result = hands.process(rgb_frame)
    frame_height, frame_width, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get index finger tip position
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Get finger base positions (for checking clenched fingers)
            thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
            index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                                                        
            # Convert to pixel coordinates
            x = int(index_finger_tip.x * frame_width)
            y = int(index_finger_tip.y * frame_height)

            # Convert coordinates to screen size
            screen_x = np.interp(x, [0, frame_width], [0, screen_width])
            screen_y = np.interp(y, [0, frame_height], [0, screen_height])

            # Get current cursor position before interpolation
            current_x, current_y = pyautogui.position()

            # Dynamic smoothing for cursor movement
            smooth_factor = 7 if abs(current_x - screen_x) > 50 or abs(current_y - screen_y) > 50 else 5
            new_x = (current_x * (smooth_factor - 1) + screen_x) / smooth_factor
            new_y = (current_y * (smooth_factor - 1) + screen_y) / smooth_factor
            pyautogui.moveTo(new_x, new_y, duration=0.05)  # Added a tiny delay for natural movement




            # Calculate distances between fingers
            thumb_index_dist = np.hypot((index_finger_tip.x - thumb_tip.x), (index_finger_tip.y - thumb_tip.y))
            thumb_middle_dist = np.hypot((middle_finger_tip.x - thumb_tip.x), (middle_finger_tip.y - thumb_tip.y))
            index_middle_dist = np.hypot((middle_finger_tip.x - index_finger_tip.x), (middle_finger_tip.y - index_finger_tip.y))
            ring_pinky_dist = np.hypot((ring_finger_tip.x - pinky_finger_tip.x), (ring_finger_tip.y - pinky_finger_tip.y))
            middle_ring_dist = np.hypot((middle_finger_tip.x - ring_finger_tip.x), (middle_finger_tip.y - ring_finger_tip.y))

            # Get clenched state of each finger
            thumb_clenched = thumb_tip.y > thumb_base.y
            index_clenched = index_finger_tip.y > index_base.y
            middle_clenched = middle_finger_tip.y > middle_base.y
            ring_clenched = ring_finger_tip.y > ring_base.y
            pinky_clenched = pinky_finger_tip.y > pinky_base.y

            # Scroll distance
            scroll_threshold = 25
            zoom_threshold = 0.02  # Adjust this value to change zoom sensitivity


            # Left Click: Thumb & Index Finger Pinch
            # Left Click: Three fingers clenched (pinky, ring, and middle)
            if pinky_clenched and ring_clenched and middle_clenched and not index_clenched and not thumb_clenched and not left_click_active:
                if (current_time - last_click_time) > click_cooldown:
                    if (current_time - last_click_time) < double_click_threshold:
                        pyautogui.doubleClick()
                    else:
                        pyautogui.click()
                    last_click_time = current_time
                left_click_active = True  # Prevent multiple clicks
            else:
                left_click_active = False


            # Right Click: Thumb & Middle Finger Pinch
            if thumb_middle_dist < 0.05 and not right_click_active:
                if (current_time - last_click_time) > right_click_cooldown:
                    pyautogui.rightClick()
                    last_click_time = current_time  # Update time to prevent immediate repeat
                right_click_active = True
            elif thumb_middle_dist >= 0.05:
                right_click_active = False

            # Scroll: Move middle finger up/down while index is close
            if pinky_clenched and ring_clenched and thumb_clenched and not index_clenched and not middle_clenched:
                if index_middle_dist < 0.04:
                    if previous_y is not None:
                        y_movement = middle_finger_tip.y - previous_y
                        scroll_speed = int(abs(y_movement) * 500) # Adjust scroll speed
                        if abs(y_movement) > 0.003:  # Threshold to prevent accidental scrolling
                            scroll_amount = int(y_movement * 300)
                            pyautogui.scroll(-scroll_amount)
                    previous_y = middle_finger_tip.y
                else:
                    previous_y = None  # Reset when not scrolling


        # Zoom with thumb and index finger pinch
        # Zoom: When all fingers except thumb and index are clenched
        if pinky_clenched and ring_clenched and  not middle_clenched and not index_clenched and not thumb_clenched:
            if thumb_index_dist < 0.05 and previous_zoom_distance is not None:  # Adjust threshold for sensitivity
                zoom_change = thumb_index_dist - previous_zoom_distance
                if abs(zoom_change) > zoom_threshold:  # Only apply zoom if change is significant
                    if zoom_change > 0:
                        pyautogui.hotkey('ctrl', '+')
                    else:
                        pyautogui.hotkey('ctrl', '-')
            previous_zoom_distance = thumb_index_dist
        else:
            previous_zoom_distance = None  # Reset when not zooming


            # Draw hand landmarks
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow("Virtual Mouse", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
