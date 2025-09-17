from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import tensorflow as tf

app = Flask(__name__)

# Load TFLite model (change to model_quant.tflite if using quantized)
interpreter = tf.lite.Interpreter(model_path="model_quant.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input tensor info
input_dtype = input_details[0]['dtype']
input_scale, input_zero_point = input_details[0]['quantization'] if input_dtype == np.uint8 else (1.0, 0)

# Output tensor info
output_dtype = output_details[0]['dtype']
output_scale, output_zero_point = output_details[0]['quantization'] if output_dtype == np.uint8 else (1.0, 0)

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get base64 image from request
        data = request.json['image']
        image_bytes = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Preprocess image
        img = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0

        # Quantize if needed
        if input_dtype == np.uint8:
            img = img / input_scale + input_zero_point
            img = np.clip(img, 0, 255).astype(np.uint8)

        img = np.expand_dims(img, axis=0)

        # Set tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        # Get predictions
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Dequantize if needed
        if output_dtype == np.uint8:
            predictions = (predictions.astype(np.float32) - output_zero_point) * output_scale

        # Get predicted label
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
    app.run(debug=True)
