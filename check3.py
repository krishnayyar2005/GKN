import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import time
import threading
import logging

logging.getLogger('mediapipe').setLevel(logging.ERROR)

DATASET_DIR = ".asl_dataset/asl_alphabet_train/asl_alphabet_train"  
MODEL_PATH = "asl_knn_model.pkl"

# === NORMALIZATION ===
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]  # wrist as origin
    normalized = landmarks - base
    return normalized.flatten()

# === LANDMARK EXTRACTION ===
def extract_landmarks_from_image(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            return normalize_landmarks(landmarks)
    return None

# === LOAD DATASET ===
def load_dataset(max_per_class=100):
    print("📂 Loading and processing dataset...")
    data, labels = [], []

    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Dataset not found at {DATASET_DIR}")
        return data, labels

    letter_folders = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

    for label in letter_folders:
        folder_path = os.path.join(DATASET_DIR, label)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:max_per_class]
        print(f"⏳ {label}: {len(image_files)} images")

        count = 0
        for fname in image_files:
            image = cv2.imread(os.path.join(folder_path, fname))
            if image is None: continue

            landmarks = extract_landmarks_from_image(image)
            if landmarks is not None and len(landmarks) == 63:
                data.append(landmarks)
                labels.append(label)
                count += 1

        print(f"   ✓ Processed {count} for {label}")

    print(f"✅ Loaded {len(data)} samples")
    return data, labels

# === TRAIN MODEL ===
def train_model(data, labels):
    print("🧠 Training KNN model...")
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(data, labels)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {MODEL_PATH}")
    return model

# === LIVE PREDICTION ===
def live_prediction(model):
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    prediction_data = {"label": "Waiting...", "confidence": 0.0, "running": True}
    frame_data = {"frame": None}

    def prediction_thread():
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
            while prediction_data["running"]:
                if frame_data["frame"] is not None:
                    rgb = cv2.cvtColor(frame_data["frame"], cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        landmarks = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            landmarks.append([lm.x, lm.y, lm.z])
                        normalized = normalize_landmarks(landmarks)
                        pred_probs = model.predict_proba([normalized])[0]
                        max_idx = np.argmax(pred_probs)

                        # Optional: print top-3 predictions for debug
                        top3_idx = np.argsort(pred_probs)[::-1][:3]
                        print("🔎 Top Predictions:", [(model.classes_[i], round(pred_probs[i], 2)) for i in top3_idx])

                        if pred_probs[max_idx] >= 0.75:
                            prediction_data["label"] = model.classes_[max_idx]
                            prediction_data["confidence"] = pred_probs[max_idx]
                        else:
                            prediction_data["label"] = "Uncertain"
                            prediction_data["confidence"] = pred_probs[max_idx]
                    else:
                        prediction_data["label"] = "No hand"
                        prediction_data["confidence"] = 0.0
                time.sleep(0.03)

    threading.Thread(target=prediction_thread, daemon=True).start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Webcam not available")
        prediction_data["running"] = False
        return

    print("🎥 Live Prediction (Press 'q' to exit)")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame_data["frame"] = frame.copy()

            with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            label_text = f"{prediction_data['label']} ({prediction_data['confidence']:.2f})"
            cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("ASL Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        prediction_data["running"] = False
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Session ended")

# === MAIN ===
def main():
    try:
        if os.path.exists(MODEL_PATH):
            print(f"📦 Loading model from {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        else:
            data, labels = load_dataset()
            if not data:
                print("❌ No training data found")
                return
            model = train_model(data, labels)

        live_prediction(model)

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
