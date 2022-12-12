import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)
# a = 0
cap = cv2.VideoCapture(0)
print(cap)
while True:
    ret, frame = cap.read()
    image_height, image_width, _ = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 因为摄像头是镜像的，所以将摄像头水平翻转
    # 不是镜像的可以不翻转
    frame = cv2.flip(frame, 2)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_handedness:
        for hand_label in results.multi_handedness:
            pass
            # print(hand_label)
    if results.multi_hand_landmarks:
        # for hand_landmarks in results.multi_hand_landmarks:
        #   print('hand_landmarks:',hand_landmarks)
        #   # 关键点可视化
        #   mp_drawing.draw_landmarks(
        #       frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        for hand_landmarks in results.multi_hand_landmarks:
            # 关键点可视化
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            print(
                f'#3: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y})\n'
                f'#4: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y})\n'
                f'#5: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y})\n'
                f'#6: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y})\n'
                f'#7: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y})\n'
                f'#8: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y})\n'
                f'#9: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y})\n'
                f'#11: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y})\n'
                f'#12: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y})\n'
                f'#13: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y})\n'
                f'#15: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y})\n'
                f'#16: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y})\n'
                f'#17: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y})\n'
                f'#19: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y})\n'
                f'#20: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y})\n'
            )
            one = 0
            two = 0
            three = 0
            four = 0
            five = 0
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_MCP].y:
                text = "one"
                one = 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_MCP].y:
                text = "two"
                two = 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y:
                text = "three"
                three = 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[
                mp_hands.HandLandmark.RING_FINGER_MCP].y:
                text = "four"
                four = 1
            # if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[
            #     mp_hands.HandLandmark.THUMB_IP].y:
            #     text = "five"
            #     five = 1
            if one == 0 and two == 0 and three == 0 and four == 0 and five == 0:
                text = "zero"
    else:
        text = "null"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
