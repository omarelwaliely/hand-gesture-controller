import ctypes
import threading
import cv2
import numpy as np
import mediapipe as mp
import pydirectinput

class HandGestureControl:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, movement_multiplier=2.5, required_consecutive_frames=10):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.screen_width, self.screen_height = pydirectinput.size()
        self.toggled_keys = {}
        self.gesture_cooldowns = {}
        self.frame_count = 0
        self.active_key = None
        self.consecutive_gesture = None
        self.consecutive_count = 0
        self.required_consecutive_frames = required_consecutive_frames
        self.movement_multiplier = movement_multiplier
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            max_num_hands=max_num_hands,
            min_tracking_confidence=min_tracking_confidence,
            min_detection_confidence=min_detection_confidence
        )
        self.gesture_key_map = {
        'forward': ('w', 15),  #thumbs up
        'backward': ('s', 15),  #thumbs down
        'left click': ('left_click', 15), #index up
        'scroll down': ('scroll_down', 10), #index pink up
        'toggle jump': (['space'], -5), #ok sign
        'right click': ('right_click', 5), #index,middle,pinky up
        'scroll up':('scroll_up', 10), #pinky up
        'toggle sprint':(['r'],-5), # three sign (index,ring, middle up)
        'inventory':('e',10),#index middle pinky up
        'toggle crouch': ('k',-5) #index ring pinky
    }

    def press_key_once(self, key):
        pydirectinput.press(key)

    def hold_key(self, key):
        if self.active_key != key:
            if self.active_key:
                if self.active_key == 'right_click':
                    pydirectinput.mouseUp(button='right')
                elif self.active_key == 'left_click':
                    pydirectinput.mouseUp(button='left')
                else:
                    pydirectinput.keyUp(self.active_key)
                print(f"{self.active_key} is being released")
            print(f"{key} is being held down")
            if key == 'right_click':
                pydirectinput.mouseDown(button='right')
            elif key == 'left_click':
                pydirectinput.mouseDown(button='left')
            else:
                pydirectinput.keyDown(key)
            self.active_key = key

    def release_key(self):
        if self.active_key:
            print(f"{self.active_key} is being released")
            if self.active_key == 'right_click':
                pydirectinput.mouseUp(button='right')
            elif self.active_key == 'left_click':
                pydirectinput.mouseUp(button='left')
            else:
                pydirectinput.keyUp(self.active_key)
            self.active_key = None

    def toggle_keys(self, keys):
        self.release_key()
        for key in keys:
            print(key)
            if self.toggled_keys.get(key, False):
                pydirectinput.keyUp(key)
                self.toggled_keys[key] = False
            else:
                pydirectinput.keyDown(key)
                self.toggled_keys[key] = True

    def move_mouse(self, hand_landmarks):
        if hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            hand_x = 1 - index_tip.x
            hand_y = index_tip.y
            adjusted_x = (hand_x - 0.5) * self.movement_multiplier + 0.5
            adjusted_y = (hand_y - 0.5) * self.movement_multiplier + 0.5
            adjusted_x = max(0, min(1, adjusted_x))
            adjusted_y = max(0, min(1, adjusted_y))
            target_x = int(adjusted_x * self.screen_width)
            target_y = int(adjusted_y * self.screen_height)
            current_x, current_y = pydirectinput.position()
            smooth_factor = 0.05
            new_x = int(current_x + (target_x - current_x) * smooth_factor)
            new_y = int(current_y + (target_y - current_y) * smooth_factor)
            pydirectinput.moveTo(new_x, new_y)

    def detect_thumbs_down(self,hand_landmarks):
        if hand_landmarks:
            landmarks = hand_landmarks.landmark
            return (landmarks[4].y > landmarks[3].y) and all(landmarks[i].y < landmarks[3].y for i in [8, 12, 16, 20])
        return False

    def detect_ok_sign(self,hand_landmarks):
        if hand_landmarks:
            landmarks = hand_landmarks.landmark
            thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
            index_tip = np.array([landmarks[8].x, landmarks[8].y])
            thumb_index_distance = np.linalg.norm(thumb_tip - index_tip)
            middle_extended = landmarks[12].y < landmarks[10].y
            ring_extended = landmarks[16].y < landmarks[14].y
            pinky_extended = landmarks[20].y < landmarks[18].y
            thumb_index_close = thumb_index_distance < 0.05

            return thumb_index_close and middle_extended and ring_extended and pinky_extended

        return False

    def detect_index_ring_pinky_up(self,hand_landmarks):
        if hand_landmarks:
            landmarks = hand_landmarks.landmark
            return (landmarks[8].y < landmarks[6].y and 
                    landmarks[16].y < landmarks[14].y and  
                    landmarks[20].y < landmarks[18].y and 
                    all(landmarks[i].y > landmarks[14].y for i in [4, 12]))  
        return False

    def detect_pinky_up(self,hand_landmarks):
        if hand_landmarks:
            landmarks = hand_landmarks.landmark
            return (landmarks[20].y < landmarks[18].y) and all(
                landmarks[i].y > landmarks[18].y for i in [4, 8, 12, 16]
            )
        return False

    def detect_index_middle_pinky_up(self,hand_landmarks):
        if hand_landmarks:
            landmarks = hand_landmarks.landmark
            return (landmarks[8].y < landmarks[6].y and 
                    landmarks[12].y < landmarks[10].y and 
                    landmarks[20].y < landmarks[18].y and
                    all(landmarks[i].y > landmarks[10].y for i in [4, 16]))
        return False

    def detect_three_sign(self,hand_landmarks):
        if hand_landmarks:
            landmarks = hand_landmarks.landmark

            index_up = landmarks[8].y < landmarks[6].y
            middle_up = landmarks[12].y < landmarks[10].y
            ring_up = landmarks[16].y < landmarks[14].y

            ring_lowest = max(landmarks[16].y, landmarks[14].y)

            pinky_down = landmarks[20].y > ring_lowest

            thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
            pinky_tip = np.array([landmarks[20].x, landmarks[20].y])
            thumb_pinky_distance = np.linalg.norm(thumb_tip - pinky_tip)

            thumb_close_to_pinky = thumb_pinky_distance < 0.1  

            return index_up and middle_up and ring_up and pinky_down and thumb_close_to_pinky
        return False



    def detect_index_finger_up(self,hand_landmarks):
        if hand_landmarks:
            landmarks = hand_landmarks.landmark
            return (landmarks[8].y < landmarks[6].y and landmarks[6].y < landmarks[5].y) and \
                (landmarks[12].y > landmarks[9].y and landmarks[16].y > landmarks[13].y and landmarks[20].y > landmarks[17].y)
        
        return False

    def detect_index_finger_down(self,hand_landmarks):
        if hand_landmarks:
            landmarks = hand_landmarks.landmark

            index_tip_y = landmarks[8].y
            other_finger_tips_y = [landmarks[i].y for i in [4, 12, 16, 20]] 

            return (index_tip_y > landmarks[6].y) and all(index_tip_y > y for y in other_finger_tips_y)

        return False

    def detect_index_pinky_up(self,hand_landmarks):
        if hand_landmarks:
            landmarks = hand_landmarks.landmark
            return (landmarks[8].y < landmarks[6].y and landmarks[20].y < landmarks[18].y) and all(
                landmarks[i].y > landmarks[6].y for i in [4, 12, 16])
        return False

    def detect_index_middle_up(self,hand_landmarks):
        if hand_landmarks:
            landmarks =hand_landmarks.landmark
            return (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y) and all(
                landmarks[i].y > landmarks[6].y for i in [4, 16, 20])
        return False

    def detect_thumbs_up(self,hand_landmarks):
        if hand_landmarks:
            landmarks = hand_landmarks.landmark
            return (landmarks[4].y < landmarks[3].y) and all(landmarks[i].y > landmarks[3].y for i in [8, 12, 16, 20])
        return False

    def detect_gesture(self, hand_landmarks):
        if self.detect_pinky_up(hand_landmarks):
            return "scroll down"
        elif self.detect_index_ring_pinky_up(hand_landmarks):
            return "toggle crouch"
        elif self.detect_index_pinky_up(hand_landmarks):
            return "scroll up"
        elif self.detect_ok_sign(hand_landmarks):
            return "toggle jump"
        elif self.detect_thumbs_down(hand_landmarks):
            return "backward"
        elif self.detect_thumbs_up(hand_landmarks):
            return "forward"
        elif self.detect_index_middle_pinky_up(hand_landmarks):
            return "inventory"
        elif self.detect_index_middle_up(hand_landmarks):
            return "left click"
        elif self.detect_three_sign(hand_landmarks):
            return "toggle sprint"
        elif self.detect_index_finger_up(hand_landmarks):
            return "right click" 
        return None

    def perform_scroll(self,direction):
        self.release_key()
        MOUSEEVENTF_WHEEL = 0x0800
        if(direction == 'scroll_down'):
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, 120, 0)
        else:
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -120, 0)
            
    def handle_key_action(self,key, cooldown):
        if cooldown < 0:
            self.toggle_keys(key)
        elif 'scroll' in key:
            self.perform_scroll(key)
        elif key:
            self.hold_key(key)

    def process_frame(self,frame):
        detected_gesture = "None"
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks or not results.multi_handedness:
            self.consecutive_gesture = None
            self.consecutive_count = 0
            self.release_key()
            return frame

        hand_data = list(zip(results.multi_hand_landmarks, results.multi_handedness))
        hand_data.sort(key=lambda x: x[1].classification[0].label == "Right", reverse=True)
        
        gesture_detected = None
        for idx, (hand_landmarks, _) in enumerate(hand_data):
            if idx == 0:
                gesture = self.detect_gesture(hand_landmarks)
                
                if gesture:
                    detected_gesture = gesture

                    # Check if it's the same gesture as last frame
                    if gesture == self.consecutive_gesture:
                        self.consecutive_count += 1
                    else:
                        self.consecutive_gesture = gesture
                        self.consecutive_count = 1
                    
                    if self.consecutive_count >= self.required_consecutive_frames:
                        if gesture in self.gesture_key_map:
                            key, cooldown = self.gesture_key_map[gesture]
                            new_cooldown = cooldown if cooldown > 0 else cooldown * -1
                            last_activation = self.gesture_cooldowns.get(gesture, -new_cooldown)
                            if self.frame_count - last_activation >= new_cooldown:
                                threading.Thread(target=self.handle_key_action, args=(key, cooldown), daemon=True).start()
                                self.gesture_cooldowns[gesture] = self.frame_count
                        gesture_detected = True
                else:
                    self.consecutive_gesture = None
                    self.consecutive_count = 0

            elif idx == 1:
                threading.Thread(target=self.move_mouse, args=(hand_landmarks), daemon=True).start()

            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if not gesture_detected:
            self.release_key()

        cv2.putText(frame, f"Gesture: {detected_gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        self.frame_count += 1
        return frame

    
    def start(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            frame = self.process_frame(frame)
            cv2.imshow('Gesture Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
