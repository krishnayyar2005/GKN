import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import time

KNN_MODEL_PATH = "asl_knn_model.pkl"
MLP_MODEL_PATH = "asl_mlp_model.pkl"
SCALER_PATH = "asl_scaler.pkl"

def live_prediction(knn_model, mlp_model, scaler):
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    prediction_data = {"label": "Waiting...", "confidence": 0.0, "running": True}
    frame_data = {"frame": None}

    def prediction_thread():
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
            while prediction_data["running"]:
                if frame_data["frame"] is not None:
                    frame = frame_data["frame"].copy()
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    if results.multi_hand_landmarks:
                        landmarks = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])

                        if len(landmarks) == 63:
                            scaled = scaler.transform([landmarks])
                            knn_probs = knn_model.predict_proba(scaled)[0]
                            mlp_probs = mlp_model.predict_proba(scaled)[0]
                            combined_probs = (knn_probs + mlp_probs) / 2
                            max_idx = np.argmax(combined_probs)
                            label = knn_model.classes_[max_idx]
                            confidence = combined_probs[max_idx]
                            prediction_data["label"] = label
                            prediction_data["confidence"] = confidence
                        else:
                            prediction_data["label"] = "Invalid landmarks"
                            prediction_data["confidence"] = 0.0
                    else:
                        prediction_data["label"] = "No hand detected"
                        prediction_data["confidence"] = 0.0
                time.sleep(0.03)

    pred_thread = threading.Thread(target=prediction_thread)
    pred_thread.daemon = True
    pred_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        prediction_data["running"] = False
        return

    print("🎥 Starting live prediction. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame_data["frame"] = frame.copy()

            with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            label_text = f"{prediction_data['label']}"
            if prediction_data["confidence"] > 0:
                label_text += f" ({prediction_data['confidence']:.2f})"

            cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("ASL Hand Sign Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        prediction_data["running"] = False
        cap.release()
        cv2.destroyAllWindows()
        print("👋 ASL Recognition stopped")

def main():
    with open(KNN_MODEL_PATH, "rb") as f:
        knn_model = pickle.load(f)
    with open(MLP_MODEL_PATH, "rb") as f:
        mlp_model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    live_prediction(knn_model, mlp_model, scaler)

if __name__ == "__main__":
    main()
