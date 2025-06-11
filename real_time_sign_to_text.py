import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# Load model and label map
model = joblib.load(r"D:\Python_Program\BPUT Hackthon\Project_Experiment\gesture_landmark_model.pkl")
label_map = np.load(r"D:\Python_Program\BPUT Hackthon\Project_Experiment\label_map.npy", allow_pickle=True).item()
reverse_map = {v: k for k, v in label_map.items()}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
prev_letter = ""
start_time = 0
output_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    landmark_list = []

    if result.multi_hand_landmarks:
        # Collect up to 2 hands' landmarks
        hand_landmarks_all = result.multi_hand_landmarks[:2]

        for hand_landmarks in hand_landmarks_all:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

        # Pad with zeros if fewer than 126 features
        while len(landmark_list) < 126:
            landmark_list.extend([0.0, 0.0, 0.0])

        if len(landmark_list) == 126:
            prediction = model.predict([landmark_list])[0]

            if prediction in reverse_map:
                current_letter = reverse_map[prediction]
            else:
                current_letter = "?"  # Unknown label

            if current_letter != "?":
                if current_letter == prev_letter:
                    if time.time() - start_time >= 3:  # Wait for 3 seconds
                        if current_letter == "SPACE":
                            output_text += "_"
                        else:
                            output_text += current_letter
                        start_time = time.time()
                else:
                    prev_letter = current_letter
                    start_time = time.time()

            # Show current letter
            display_letter = "_" if current_letter == "SPACE" else current_letter
            cv2.putText(image, f"Letter: {display_letter}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # Blinking cursor logic
    cursor_char = "|" if int(time.time() * 2) % 2 == 0 else " "
    display_output = output_text + cursor_char

    # Show final output
    cv2.putText(image, f"Output: {display_output}", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Sign to Text", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
