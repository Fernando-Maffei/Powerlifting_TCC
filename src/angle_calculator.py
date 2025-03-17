import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


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


# Inicialização do MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Pasta raiz onde os vídeos estão armazenados
pasta_raiz = "dataset/Exercicios"

# Lista para armazenar todos os dados extraídos
data = []

# Percorre todos os vídeos na pasta raiz e subpastas
for root, _, files in os.walk(pasta_raiz):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            print(f"Processando vídeo: {video_path}")

            # Abre o vídeo
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Erro ao abrir o vídeo: {video_path}")
                continue

            # Processa cada frame do vídeo
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(frame_rgb)

                if result.pose_landmarks:
                    keypoints = extract_keypoints(
                        result.pose_landmarks.landmark)
                    data.append(keypoints)

            cap.release()

# Verifica se há dados antes de aplicar o StandardScaler
if data:
    data = np.array(data)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print("Dados normalizados com sucesso.")
else:
    print("Nenhum dado foi extraído. Verifique os vídeos ou a detecção de poses.")
