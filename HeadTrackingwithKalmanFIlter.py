import cv2
import mediapipe as mp
import numpy as np
import math
import time
from scipy.spatial.distance import euclidean

# Konfiguracja dla Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)

# Funkcja obliczająca odległość euklidesową
def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Inicjalizacja filtru Kalmana
def create_kalman_filter():
    kalman = cv2.KalmanFilter(6, 3)  # 6 stanów (x, y, z, vx, vy, vz), 3 obserwacje (x, y, z)
    kalman.measurementMatrix = np.eye(3, 6, dtype=np.float32)
    kalman.transitionMatrix = np.array([
        [1, 0, 0, 1, 0, 0],  # x' = x + vx
        [0, 1, 0, 0, 1, 0],  # y' = y + vy
        [0, 0, 1, 0, 0, 1],  # z' = z + vz
        [0, 0, 0, 1, 0, 0],  # vx' = vx
        [0, 0, 0, 0, 1, 0],  # vy' = vy
        [0, 0, 0, 0, 0, 1],  # vz' = vz
    ], dtype=np.float32)
    kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.005  # Mniejszy szum procesowy
    kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.02  # Mniejszy szum pomiarowy
    kalman.errorCovPost = np.eye(6, dtype=np.float32)
    kalman.statePost = np.zeros((6, 1), dtype=np.float32)
    return kalman

kalman_filter = create_kalman_filter()

# Przechwytywanie obrazu z kamery
cap = cv2.VideoCapture(0)
last_print_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Odbicie lustrzane obrazu
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wykrywanie twarzy za pomocą Mediapipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Landmarki dla kluczowych punktów
            nose_landmark = face_landmarks.landmark[1]
            left_eye_landmark = face_landmarks.landmark[33]
            right_eye_landmark = face_landmarks.landmark[133]
            left_cheek_landmark = face_landmarks.landmark[234]
            right_cheek_landmark = face_landmarks.landmark[454]
            chin_landmark = face_landmarks.landmark[152]
            forehead_landmark = face_landmarks.landmark[10]

            # Współrzędne (x, y) kluczowych punktów
            nose = (int(nose_landmark.x * w), int(nose_landmark.y * h))
            left_eye = (int(left_eye_landmark.x * w), int(left_eye_landmark.y * h))
            right_eye = (int(right_eye_landmark.x * w), int(right_eye_landmark.y * h))
            left_cheek = (int(left_cheek_landmark.x * w), int(left_cheek_landmark.y * h))
            right_cheek = (int(right_cheek_landmark.x * w), int(right_cheek_landmark.y * h))
            chin = (int(chin_landmark.x * w), int(chin_landmark.y * h))
            forehead = (int(forehead_landmark.x * w), int(forehead_landmark.y * h))

            # Obliczanie odległości
            eye_distance = euclidean(left_eye, right_eye)
            face_height = euclidean(chin, forehead)
            face_width = euclidean(left_cheek, right_cheek)

            # Uśrednianie pomiaru
            dominant_distance = max(eye_distance, face_height, face_width)
            avg_face_size = dominant_distance

            # Normalizacja do zakresu 0-200 dla z
            z = np.interp(avg_face_size, [50, 300], [200, 0])
            z = max(0, min(200, z))


            # Korekta i predykcja Kalmana
            measurement = np.array([[np.float32(nose[0])], [np.float32(nose[1])], [np.float32(z)]], dtype=np.float32)
            kalman_filter.correct(measurement)  # Korekta
            prediction = kalman_filter.predict()  # Predykcja

            # Rysowanie punktów
            cv2.circle(frame, left_cheek, 3, (255, 0, 0), -1)
            cv2.circle(frame, right_cheek, 3, (255, 0, 0), -1)
            cv2.circle(frame, chin, 3, (0, 255, 0), -1)
            cv2.circle(frame, forehead, 3, (0, 255, 0), -1)
            cv2.circle(frame, (int(prediction[0][0]), int(prediction[1][0])), 5, (0, 0, 255), -1)

            # Wyświetlanie pozycji
            current_time = time.time()
            if current_time - last_print_time >= 0.5:
                print(f"Estymacja: x={nose[0]:.2f}, y={nose[1]:.2f}, z={prediction[2][0]:.2f}")
                last_print_time = current_time

    # Wyświetlanie obrazu
    cv2.imshow("Head Tracking with Kalman Filter", frame)

    # Zakończenie programu przy naciśnięciu ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
