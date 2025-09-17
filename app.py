from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
import os

app = Flask(__name__)

# Load Keras model
model = load_model("model.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.json['image']
        image_bytes = base64.b64decode(data.split(',')[1])  # remove header
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Preprocess
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

        # Predict
        predictions = model.predict(img)[0]
        max_index = np.argmax(predictions)
        confidence = float(predictions[max_index])
        label = labels[max_index]

        return jsonify({
            "label": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
