import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import joblib
from angle_calculator import calculate_angle

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class PoseDetector:
    def __init__(self, model_paths):
        """
        Inicializa o detector de poses com modelos para diferentes movimentos.
        """
        self.models = {
            "Agachamento": joblib.load(model_paths["Agachamento"]),
            "Supino": joblib.load(model_paths["Supino"]),
            "Levantamento Terra": joblib.load(model_paths["Levantamento Terra"])
        }
        self.pose = mp.solutions.pose.Pose()

    def process_frame(self, frame):
        """
        Processa um frame para detectar poses.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        return results

    def classify_movement(self, results, movement_type):
        """
        Classifica o movimento com base nos landmarks detectados.
        """
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            if movement_type == "Agachamento":
                quadril = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y)
                joelho = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y)
                tornozelo = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y)
                angulo_joelho = calculate_angle(quadril, joelho, tornozelo)
                return angulo_joelho < 90  # Verifica se o joelho está abaixo do paralelo
            elif movement_type == "Levantamento Terra":
                quadril = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y)
                joelho = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y)
                ombro = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y)
                angulo_quadril = calculate_angle(joelho, quadril, ombro)
                return angulo_quadril > 170  # Verifica se os quadris estão estendidos
        return False

class PoseDetectionApp:
    def __init__(self, root, model_paths):
        """
        Inicializa a aplicação de detecção de poses.
        """
        self.root = root
        self.root.title("Análise de Movimentos - Powerlifting")
        self.detector = PoseDetector(model_paths)
        self.is_running = False
        self.cap = None
        self.video_path = None
        self.using_camera = False
        self.setup_ui()

    def setup_ui(self):
        """
        Configura a interface gráfica da aplicação.
        """
        self.video_frame = ttk.Label(self.root)
        self.video_frame.pack(pady=10)

        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=10)

        buttons = [
            ("Selecionar Vídeo", self.select_video),
            ("Iniciar", self.start_detection, tk.DISABLED),
            ("Usar Câmera", self.start_camera),
            ("Parar", self.stop_detection, tk.DISABLED)
        ]

        self.button_refs = []
        for text, command, *state in buttons:
            btn = ttk.Button(self.button_frame, text=text, command=command)
            if state:
                btn.config(state=state[0])
            btn.pack(side=tk.LEFT, padx=5)
            self.button_refs.append(btn)

        self.movement_type = ttk.Combobox(self.button_frame, values=["Agachamento", "Supino", "Levantamento Terra"])
        self.movement_type.current(0)
        self.movement_type.pack(side=tk.LEFT, padx=5)

        self.result_label = ttk.Label(
            self.root, text="Resultados aparecerão aqui...", font=("Arial", 12))
        self.result_label.pack(pady=10)

    def select_video(self):
        """
        Abre um seletor de arquivos para escolher um vídeo.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Arquivos de Vídeo", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.video_path = file_path
            self.using_camera = False
            self.button_refs[1].config(state=tk.NORMAL)
        else:
            messagebox.showwarning("Aviso", "Nenhum vídeo selecionado.")

    def start_detection(self):
        """
        Inicia a detecção de poses no vídeo selecionado.
        """
        if not self.is_running and self.video_path:
            self.is_running = True
            self.cap = cv2.VideoCapture(self.video_path)
            self.using_camera = False
            if not self.cap.isOpened():
                messagebox.showerror("Erro", "Erro ao carregar o vídeo!")
                self.is_running = False
                return
            self.toggle_buttons(False, True)
            self.update_frame()

    def start_camera(self):
        """
        Inicia a detecção de poses usando a câmera ao vivo.
        """
        if not self.is_running:
            self.is_running = True
            self.cap = cv2.VideoCapture(0)
            self.using_camera = True
            if not self.cap.isOpened():
                messagebox.showerror("Erro", "Erro ao acessar a câmera!")
                self.is_running = False
                return
            self.toggle_buttons(False, True)
            self.update_frame()

    def stop_detection(self):
        """
        Para a detecção de poses e libera os recursos.
        """
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_frame.config(image='')
        self.toggle_buttons(True, False)

    def update_frame(self):
        """
        Atualiza o frame do vídeo ou da câmera e processa a detecção de poses.
        """
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_detection()
                return
            results = self.detector.process_frame(frame)
            annotated_frame, feedback = self.draw_landmarks(frame, results)
            img = ImageTk.PhotoImage(image=Image.fromarray(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)))
            self.video_frame.config(image=img)
            self.video_frame.image = img
            self.result_label.config(text=feedback)
            self.root.after(10, self.update_frame)

    def draw_landmarks(self, image, results):
        """
        Desenha os landmarks da pose no frame e fornece feedback.
        """
        feedback = "Aguardando detecção..."
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            movement_type = self.movement_type.get()
            is_valid = self.detector.classify_movement(results, movement_type)
            feedback = f"{movement_type}: {'Válido' if is_valid else 'Inválido'}"
        return image, feedback

    def toggle_buttons(self, enable_start, enable_stop):
        """
        Ativa ou desativa os botões conforme o estado da aplicação.
        """
        self.button_refs[1].config(
            state=tk.NORMAL if enable_start else tk.DISABLED)
        self.button_refs[0].config(
            state=tk.NORMAL if enable_start else tk.DISABLED)
        self.button_refs[2].config(
            state=tk.NORMAL if enable_start else tk.DISABLED)
        self.button_refs[3].config(
            state=tk.NORMAL if enable_stop else tk.DISABLED)

    def run(self):
        """
        Inicia a execução da aplicação.
        """
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """
        Fecha a aplicação corretamente.
        """
        self.stop_detection()
        self.root.destroy()


if __name__ == "__main__":
    MODEL_PATHS = {
        "Agachamento": "models/classificador_agachamento.pkl",
        "Supino": "models/classificador_supino.pkl",
        "Levantamento Terra": "models/classificador_levantamento_terra.pkl"
    }
    root = tk.Tk()
    app = PoseDetectionApp(root, MODEL_PATHS)
    app.run()