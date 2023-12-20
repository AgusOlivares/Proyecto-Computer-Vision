import cv2
import mediapipe as mp
import numpy as np
import os

# Iniciar la captura de video desde la cámara (usualmente la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Variables para el seguimiento del punto específico y el dibujo
track_point = None
track_history = []  # Historial de posiciones del punto a seguir
drawing_history = []  # Historial para el dibujo en pantalla
drawing_saved = False  # Indicador de si se guardó el dibujo

def tomar_captura(frame, drawing_history):
    # Crear una copia del frame para agregar el dibujo
    frame_con_dibujo = frame.copy()

    # Dibujar el historial de dibujo en el frame copiado
    if len(drawing_history) > 1:
        for i in range(1, len(drawing_history)):
            cv2.line(frame_con_dibujo, drawing_history[i - 1], drawing_history[i], (0, 255, 0), 2)

    # Crear la carpeta "Capturas" si no existe
    carpeta_capturas = 'Capturas'
    if not os.path.exists(carpeta_capturas):
        os.makedirs(carpeta_capturas)

    # Generar un nombre de archivo para la captura
    num_archivos = len(os.listdir(carpeta_capturas))
    nombre_archivo = f'captura_{num_archivos + 1}.png'
    ruta_archivo = os.path.join(carpeta_capturas, nombre_archivo)

    # Guardar la captura con el dibujo como imagen PNG en la carpeta "Capturas"
    cv2.imwrite(ruta_archivo, frame_con_dibujo)
    print(f"Captura con dibujo guardada como {nombre_archivo} en la carpeta 'Capturas'")


while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Convertir el frame a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar manos en el frame
    results = hands.process(frame_rgb)

    # Dibujar los landmarks de las manos si se detectan y manejar el seguimiento del punto específico
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                # Manejar el seguimiento del punto específico si track_point está configurado
                if track_point is not None and idx == track_point:
                    track_history.append((x, y))
                    if len(track_history) > 1:
                        for i in range(1, len(track_history)):
                            cv2.line(frame, track_history[i - 1], track_history[i], (0, 255, 0), 2)

    # Dibujar el historial de dibujo en la ventana si se ha presionado "y"
    if len(drawing_history) > 1 and not drawing_saved:
        for i in range(1, len(drawing_history)):
            cv2.line(frame, drawing_history[i - 1], drawing_history[i], (0, 255, 0), 2)

    # Mostrar el frame con las detecciones de manos y el dibujo del punto rastreado
    cv2.imshow('Hand Detection', frame)

    # Manejar la entrada del teclado
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('t'):
        # Comienza a dibujar
        track_point = 8  # Cambia esto al índice del landmark que deseas seguir
        track_history = []  # Limpia el historial de seguimiento cuando comienza un nuevo dibujo
        drawing_history = []  # Limpia el historial de dibujo al iniciar un nuevo dibujo

    elif key & 0xFF == ord('y'):
        if track_point is not None:
            drawing_history.extend(track_history)  # Agregar el seguimiento al dibujo
            track_history = []  # Limpiar el historial de seguimiento
        track_point = None  # Detener el seguimiento

    elif key & 0xFF == ord('g'):
        tomar_captura(frame, drawing_history)
    
        




# Liberar la captura y cerrar la ventana
cap.release()
cv2.destroyAllWindows()


