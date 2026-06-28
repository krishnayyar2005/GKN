import cv2
import base64
import requests
import time

def image_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def send_to_ollama(base64_img):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llava",  # make sure it's the correct name
        "prompt": "Which English alphabet is shown in this hand sign image?",
        "images": [base64_img],
        "stream": False
    })
    return response.json()

# Start webcam
cap = cv2.VideoCapture(0)

print("📷 Starting camera... Show a hand sign!")

last_prediction_time = 0
prediction_interval = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break


    current_time = time.time()
    if current_time - last_prediction_time > prediction_interval:
        base64_img = image_to_base64(frame)
        print("🧠 Thinking...")
        result = send_to_ollama(base64_img)
        print(f"👉 AI Says: {result}")
        last_prediction_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
