import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import StandardScaler

def calculate_angle(a, b, c):
    """Calcula o ângulo entre três pontos."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def extract_keypoints(landmarks):
    """Extrai keypoints e calcula ângulos relevantes."""
    keypoints = []
    for landmark in landmarks:
        keypoints.extend([landmark.x, landmark.y, landmark.z])

    # Cálculo de ângulos
    quadril = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y)
    joelho = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
              landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y)
    tornozelo = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y)

    angulo_joelho = calculate_angle(quadril, joelho, tornozelo)
    keypoints.append(angulo_joelho)

    return keypoints

def validate_squat(keypoints):
    """Valida o agachamento com base nos ângulos."""
    angulo_joelho = keypoints[-1]  # Último elemento é o ângulo do joelho
    return angulo_joelho < 90  # Verifica se o joelho está abaixo do paralelo

def validate_deadlift(keypoints):
    """Valida o levantamento terra com base nos ângulos."""
    quadril = (keypoints[mp.solutions.pose.PoseLandmark.LEFT_HIP.value * 3],
               keypoints[mp.solutions.pose.PoseLandmark.LEFT_HIP.value * 3 + 1])
    joelho = (keypoints[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value * 3],
              keypoints[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value * 3 + 1])
    ombro = (keypoints[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value * 3],
             keypoints[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value * 3 + 1])

    angulo_quadril = calculate_angle(joelho, quadril, ombro)
    return angulo_quadril > 170  # Verifica se os quadris estão estendidos

# Inicialização do MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture("video.mp4")

data = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        keypoints = extract_keypoints(result.pose_landmarks.landmark)
        data.append(keypoints)

        # Validação do agachamento
        if validate_squat(keypoints):
            cv2.putText(frame, "Squat Valido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Squat Invalido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Validação do levantamento terra
        if validate_deadlift(keypoints):
            cv2.putText(frame, "Deadlift Valido", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Deadlift Invalido", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

data = np.array(data)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)