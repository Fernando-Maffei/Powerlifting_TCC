# Arquivo: src/keypoints_extractor.py
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from angle_calculator import calculate_angle

mp_pose = mp.solutions.pose

def process_videos_from_folder(folder_path, frames_dir):
    """
    Processa vídeos dentro da pasta fornecida:
    - Extrai keypoints e salva em Parquet.
    - Filtra frames redundantes.
    - Salva frames convertidos para tons de cinza.
    """
    data = []
    pose = mp_pose.Pose()
    os.makedirs(frames_dir, exist_ok=True)

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                category = os.path.basename(os.path.dirname(root))  # Pega o nome da pasta pai (Agachamento, Terra, etc.)
                correct = 1 if "Certo" in root else 0  # Verifica se o vídeo está na pasta "Certo"
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Erro ao abrir o vídeo: {video_path}")
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = 0
                prev_gray = None
                frame_output_dir = os.path.join(frames_dir, category, "Certo" if correct else "Errado")
                os.makedirs(frame_output_dir, exist_ok=True)
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        diff = np.abs(prev_gray.astype(np.float32) - frame_gray.astype(np.float32))
                        if np.mean(diff) < 5:
                            continue
                    prev_gray = frame_gray

                    frame_count += 1
                    frame_filename = f"{category}_frame_{frame_count}.png"
                    cv2.imwrite(os.path.join(frame_output_dir, frame_filename), frame_gray)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)
                    keypoints = [frame_count / fps, category, correct]
                    
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        for landmark in landmarks:
                            keypoints.extend([landmark.x, landmark.y, landmark.z])
                        
                        # Validação do movimento
                        if category == "Agachamento":
                            quadril = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
                            joelho = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
                            tornozelo = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
                            angulo_joelho = calculate_angle(quadril, joelho, tornozelo)
                            keypoints.append(angulo_joelho)
                        elif category == "Terra":
                            quadril = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
                            joelho = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
                            ombro = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
                            angulo_quadril = calculate_angle(joelho, quadril, ombro)
                            keypoints.append(angulo_quadril)
                    else:
                        keypoints += [0] * (21 - len(keypoints))  # Preenche com zeros se não houver landmarks
                    data.append(keypoints)
                cap.release()
    
    colunas = ["timestamp", "exercicio", "correto"] + [f"kp_{i}" for i in range(20)] + ["angulo"]
    return pd.DataFrame(data, columns=colunas)

if __name__ == "__main__":
    dataset_dir = "dataset/Exercicios"  
    frames_dir = "dataset/processed/frames"  
    df_final = process_videos_from_folder(dataset_dir, frames_dir)
    output_path_parquet = "dataset/processed/keypoints_data.parquet"
    df_final.to_parquet(output_path_parquet, index=False)
    print(f"Dados salvos em {output_path_parquet}")