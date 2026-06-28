# ASL Hand Sign Recognition

A real-time American Sign Language (ASL) alphabet recognition system built using Python, OpenCV, MediaPipe, and Scikit-learn.

## Features

- Real-time webcam hand sign recognition
- Hand landmark detection using MediaPipe
- K-Nearest Neighbors (KNN) classifier
- Displays predicted letter with confidence score
- Saves and loads trained model

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn

## Project Structure

```
check4.py               # Main application
asl_knn_model.pkl       # Trained KNN model
asl_mlp_model.pkl       # Trained MLP model
asl_scaler.pkl          # Feature scaler
```

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python check4.py
```

## Note

This project was originally developed using Python 3.12 and the MediaPipe Solutions API.