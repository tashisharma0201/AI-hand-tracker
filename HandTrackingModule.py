import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
drawing_styles = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks, handedness, frame_shape):
    h, w = frame_shape[:2]
    fingers = []
    landmarks = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]

    # Corrected thumb logic
    if handedness == 'Right':
        thumb_up = landmarks[4][0] < landmarks[3][0]  # Right hand: thumb to the left
    else:
        thumb_up = landmarks[4][0] > landmarks[3][0]  # Left hand: thumb to the right
    fingers.append(thumb_up)

    # Other fingers (unchanged)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(landmarks[tip][1] < landmarks[pip][1])

    return sum(fingers)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_idx].classification[0].label
            count = count_fingers(hand_landmarks, handedness, frame.shape)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                drawing_styles, drawing_styles
            )

            # Display count near wrist
            wrist = (int(hand_landmarks.landmark[0].x * frame.shape[1]), 
                     int(hand_landmarks.landmark[0].y * frame.shape[0]))
            cv2.putText(frame, str(count), (wrist[0]-30, wrist[1]+40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)

    cv2.imshow('Finger Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
