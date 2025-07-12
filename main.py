import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 0, 17]
NOSE_TIP = 1

letters = [chr(i) for i in range(65, 91)]
letter_index = 0
selected_text = ""

# Previous states
prev_blink = False
prev_smile = False
prev_tilt = 0

# Thresholds (tune live)
BLINK_THRESH = 0.23  # 0.20 to 0.25 works mostly
SMILE_THRESH = 0.35  # 0.3 to 0.4 try
TILT_THRESH = 0.05   # 0.05 to 0.1 try

cap = cv2.VideoCapture(0)

print("\n✅ Ready!")
print("👉 Blink = Next Letter")
print("👉 Smile = Select Letter")
print("👉 Head Tilt Left = Backspace")
print("👉 ESC = Exit\n")

def euclidean(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def eye_aspect_ratio(eye, landmarks):
    A = euclidean(landmarks[eye[1]], landmarks[eye[5]])
    B = euclidean(landmarks[eye[2]], landmarks[eye[4]])
    C = euclidean(landmarks[eye[0]], landmarks[eye[3]])
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Eye blink ratio
        left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
        right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
        avg_ear = (left_ear + right_ear) / 2.0

        # Smile ratio
        mouth_w = euclidean(landmarks[MOUTH[0]], landmarks[MOUTH[1]])
        mouth_h = euclidean(landmarks[MOUTH[2]], landmarks[MOUTH[3]])
        mar = mouth_h / mouth_w

        # Nose X shift
        nose_x = landmarks[NOSE_TIP].x - 0.5

        # Debug print
        print(f"EAR: {avg_ear:.3f} | MAR: {mar:.3f} | Nose X: {nose_x:.3f}")

        # Blink detect
        blink = avg_ear < BLINK_THRESH
        if blink and not prev_blink:
            letter_index = (letter_index + 1) % len(letters)
            print(f"[Blink] Next Letter: {letters[letter_index]}")
        prev_blink = blink

        # Smile detect
        smile = mar > SMILE_THRESH
        if smile and not prev_smile:
            selected_text += letters[letter_index]
            print(f"[Smile] Selected: {selected_text}")
        prev_smile = smile

        # Tilt detect
        tilt_left = nose_x < -TILT_THRESH
        if tilt_left and prev_tilt == 0:
            selected_text = selected_text[:-1]
            print(f"[Tilt Left] Backspace: {selected_text}")
            prev_tilt = -1
        elif not tilt_left:
            prev_tilt = 0

        mp_drawing.draw_landmarks(
            frame, results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        )

    cv2.putText(frame, f"Letter: {letters[letter_index]}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, f"Chatbox: {selected_text}", (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    cv2.imshow("Gesture Keyboard Debug", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()