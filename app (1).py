from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load trained model and labels
def load_trained_model():
    model_path = "sign_language_model1.keras"
    labels_path = "class_indices.json"

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return None, None

    if not os.path.exists(labels_path):
        print(f"Warning: Class label file '{labels_path}' not found. Using default labels.")
        class_labels = {0: "Unknown", 1: "Unknown"}
    else:
        with open(labels_path, "r") as f:
            class_indices = json.load(f)
        class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping

    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    return model, class_labels

# Load the model and labels
model, class_labels = load_trained_model()
if model is None:
    print("Model failed to load. API will not function correctly.")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def predict_sign(image):
    """Predicts the sign language gesture from an image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    h, w, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Add padding and ensure valid cropping
            padding = 40
            x_min, y_min = max(x_min - padding, 0), max(y_min - padding, 0)
            x_max, y_max = min(x_max + padding, w), min(y_max + padding, h)

            if x_max - x_min > 10 and y_max - y_min > 10:
                hand_roi = image[y_min:y_max, x_min:x_max]
                target_size = model.input_shape[1:3]

                # Resize properly
                hand_roi = cv2.resize(hand_roi, (target_size[1], target_size[0]))
                hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                hand_roi = img_to_array(hand_roi) / 255.0  
                hand_roi = np.expand_dims(hand_roi, axis=0)

                predictions = model.predict(hand_roi)[0]
                confidence = np.max(predictions)
                predicted_class = np.argmax(predictions)
                predicted_label = class_labels.get(predicted_class, "Unknown")

                if confidence > 0.7:
                    return {"prediction": predicted_label, "confidence": float(confidence)}
                else:
                    return {"prediction": "No sign detected", "confidence": float(confidence)}
    return {"prediction": "No hand detected", "confidence": 0.0}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict sign language from an image."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image_array = np.frombuffer(file.read(), np.uint8)

    if image_array.size == 0:
        return jsonify({"error": "Invalid image file"}), 400

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Unable to decode image"}), 400

    result = predict_sign(image)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
