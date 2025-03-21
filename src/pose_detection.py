import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import joblib
from typing import List, Optional, Tuple

# Constantes
EXPECTED_FEATURES = 20  # Número de features esperadas pelo modelo

# Inicializa o MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class PoseDetectionApp:
    def __init__(self, root: tk.Tk, model_path: str):
        """
        Inicializa a aplicação de detecção de poses.
        """
        self.root = root
        self.root.title("Análise de Movimentos - Powerlifting")
        self.model = joblib.load(model_path)  # Carrega o modelo treinado
        self.is_running = False  # Controla se a detecção está ativa
        self.cap: Optional[cv2.VideoCapture] = None  # Captura de vídeo
        self.video_path: Optional[str] = None  # Caminho do vídeo selecionado
        self.using_camera = False  # Indica se a câmera está sendo usada
        self.screen_width = self.root.winfo_screenwidth()  # Largura da tela
        self.screen_height = self.root.winfo_screenheight()  # Altura da tela

        # Configura a interface gráfica
        self.setup_ui()

    def setup_ui(self):
        """
        Configura a interface gráfica da aplicação.
        """
        # Frame para exibição do vídeo
        self.video_frame = ttk.Label(self.root)
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        # Frame para os botões
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=10)

        # Botões da interface
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

        # Label para exibir o resultado
        self.result_label = ttk.Label(
            self.root, text="Resultados aparecerão aqui...",
            font=("Arial", 16, "bold"), foreground="white", background="gray"
        )
        self.result_label.pack(pady=10, fill=tk.X)

    def select_video(self):
        """
        Abre um seletor de arquivos para escolher um vídeo.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Arquivos de Vídeo", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.video_path = file_path
            self.using_camera = False
            # Ativa o botão "Iniciar"
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
            self.toggle_buttons(False, True)  # Atualiza o estado dos botões
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
            self.toggle_buttons(False, True)  # Atualiza o estado dos botões
            self.update_frame()

    def stop_detection(self):
        """
        Para a detecção de poses e libera os recursos.
        """
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_frame.config(image='')  # Limpa o frame do vídeo
        self.toggle_buttons(True, False)  # Atualiza o estado dos botões

    def update_frame(self):
        """
        Atualiza o frame do vídeo ou da câmera e processa a detecção de poses.
        """
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_detection()
                return

            # Processa o frame para detectar poses
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detect_pose(frame_rgb)

            # Desenha os landmarks e exibe o resultado
            annotated_frame, feedback = self.draw_landmarks(frame, results)

            # Redimensiona o frame proporcionalmente
            frame_resized = self.resize_frame(annotated_frame)

            # Exibe o frame na interface
            img = ImageTk.PhotoImage(image=Image.fromarray(
                cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
            self.video_frame.config(image=img)
            self.video_frame.image = img
            self.result_label.config(text=feedback)

            # Agenda a próxima atualização
            self.root.after(10, self.update_frame)

    def detect_pose(self, frame) -> mp.solutions.pose.Pose:
        """
        Detecta poses no frame usando o MediaPipe.
        """
        with mp.solutions.pose.Pose() as pose:
            results = pose.process(frame)
            return results

    def draw_landmarks(self, image, results) -> Tuple[cv2.Mat, str]:
        """
        Desenha os landmarks da pose no frame e fornece feedback.
        """
        feedback = "Aguardando detecção..."
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            feedback = self.classify_movement(
                results)  # Classifica o movimento
        return image, feedback

    def classify_movement(self, results) -> str:
        """
        Classifica o movimento com base nos landmarks detectados.
        """
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])

            # Ajusta o número de keypoints para o esperado pelo modelo
            if len(keypoints) < EXPECTED_FEATURES:
                keypoints += [0] * (EXPECTED_FEATURES - len(keypoints))
            elif len(keypoints) > EXPECTED_FEATURES:
                keypoints = keypoints[:EXPECTED_FEATURES]

            try:
                prediction = self.model.predict([keypoints])
                return "Correto" if prediction[0] == 1 else "Errado"
            except Exception as e:
                print(f"Erro ao classificar movimento: {e}")
                return "Erro na classificação"
        return "Aguardando detecção..."

    def resize_frame(self, frame) -> cv2.Mat:
        """
        Redimensiona o frame proporcionalmente para se ajustar à tela.
        """
        height, width = frame.shape[:2]
        scale = min(self.screen_width / width, self.screen_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def toggle_buttons(self, enable_start: bool, enable_stop: bool):
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
    MODEL_PATH = "models/classificador_movimentos.pkl"
    root = tk.Tk()
    app = PoseDetectionApp(root, MODEL_PATH)
    app.run()
