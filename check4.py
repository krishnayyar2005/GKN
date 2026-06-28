import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import time
import threading
import logging

# Suppress MediaPipe warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# === SETTINGS ===
DATASET_DIR = ".asl_dataset/asl_alphabet_train/asl_alphabet_train"  
MODEL_PATH = "asl_knn_model.pkl"


def extract_landmarks_from_image(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks
    return None

# === 2. DATASET LOADING ===
def load_dataset(max_per_class=100):  # Reduced sample count for faster training
    print("📂 Loading and processing dataset...")
    data = []
    labels = []
    
    # Verify directory exists
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Dataset directory not found: {DATASET_DIR}")
        return data, labels

    # Get letter folders
    letter_folders = sorted([d for d in os.listdir(DATASET_DIR) 
                           if os.path.isdir(os.path.join(DATASET_DIR, d))])
    
    if not letter_folders:
        print(f"❌ Error: No class folders found in {DATASET_DIR}")
        return data, labels
    
    print(f"🔍 Found {len(letter_folders)} classes: {', '.join(letter_folders[:5])}...")
    
    try:
        for label in letter_folders:
            folder_path = os.path.join(DATASET_DIR, label)
            processed = 0
            
            # Get image files
            image_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_per_class]
            
            print(f"⏳ Processing class {label}: {len(image_files)} images")
            
            for filename in image_files:
                img_path = os.path.join(folder_path, filename)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                    
                landmarks = extract_landmarks_from_image(image)
                if landmarks and len(landmarks) == 63:  # 21 landmarks × 3 coordinates
                    data.append(landmarks)
                    labels.append(label)
                    processed += 1
            
            print(f"   ✓ Processed {processed} images for class {label}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted! Using data collected so far...")
    
    print(f"✅ Loaded {len(data)} valid samples across {len(set(labels))} classes")
    return data, labels

# === 3. MODEL TRAINING ===
def train_model(data, labels):
    print("🧠 Training KNN model...")
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(data, labels)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model trained and saved to {MODEL_PATH}")
    
    return model

# === 4. REAL-TIME PREDICTION WITH IMPROVED THREADING ===
def live_prediction(model):
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    
    # Shared variables between threads
    prediction_data = {
        "label": "Waiting...",
        "confidence": 0.0,
        "running": True
    }
    
    # Background prediction thread
    def prediction_thread():
        # Initialize hands in the thread that uses it
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
            while prediction_data["running"]:
                if frame_data["frame"] is not None:
                    # Get a reference to current frame
                    current_frame = frame_data["frame"].copy()
                    
                    # Process frame for hand detection
                    rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    
                    if results.multi_hand_landmarks:
                        landmarks = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        
                        # Make prediction
                        pred_probs = model.predict_proba([landmarks])[0]
                        max_idx = np.argmax(pred_probs)
                        
                        # Update shared prediction data
                        prediction_data["label"] = model.classes_[max_idx]
                        prediction_data["confidence"] = pred_probs[max_idx]
                    else:
                        prediction_data["label"] = "No hand detected"
                        prediction_data["confidence"] = 0.0
                
                # Short sleep to prevent CPU overuse
                time.sleep(0.03)
    
    # Structure to share frame data
    frame_data = {"frame": None}
    
    # Start prediction thread
    pred_thread = threading.Thread(target=prediction_thread)
    pred_thread.daemon = True
    pred_thread.start()
    
    # Main thread handles camera and display
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            prediction_data["running"] = False
            return
        
        print("🎥 Starting live prediction. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Mirror the frame for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Update shared frame data
            frame_data["frame"] = frame.copy()
            
            # Process hands for visualization
            with mp_hands.Hands(
                max_num_hands=1, 
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            ) as hands:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS)
            
            # Display prediction on frame
            label_text = f"{prediction_data['label']}"
            if prediction_data["confidence"] > 0:
                label_text += f" ({prediction_data['confidence']:.2f})"
                
            cv2.putText(
                frame, 
                label_text, 
                (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Add instructions
            cv2.putText(
                frame,
                "Press 'q' to quit", 
                (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            
            # Show frame
            cv2.imshow("ASL Hand Sign Recognition", frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up resources
        prediction_data["running"] = False
        cap.release()
        cv2.destroyAllWindows()
        print("👋 ASL Recognition stopped")

# === 5. MAIN EXECUTION ===
def main():
    try:
        # Check if model exists
        if os.path.exists(MODEL_PATH):
            print(f"🔄 Loading existing model from {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully")
        else:
            print("🔄 No existing model found. Training new model...")
            data, labels = load_dataset()
            if len(data) == 0:
                print("❌ Error: No valid training data found")
                return
            model = train_model(data, labels)
        
        # Start real-time prediction
        live_prediction(model)
        
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
