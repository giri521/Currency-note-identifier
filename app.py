import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pyttsx3
import tempfile

# Initialize TTS engine
tts = pyttsx3.init()
tts.setProperty('rate', 150)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_dtype = input_details[0]['dtype']
input_scale, input_zero_point = input_details[0]['quantization'] if input_dtype == np.uint8 else (1.0, 0)
output_dtype = output_details[0]['dtype']
output_scale, output_zero_point = output_details[0]['quantization'] if output_dtype == np.uint8 else (1.0, 0)

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

st.set_page_config(page_title="Currency Identifier", page_icon="💵")

st.title("Currency Identifier")
st.write("Capture a currency note using your webcam and get predictions.")

# Webcam capture
img_file_buffer = st.camera_input("Capture Note Image")

if img_file_buffer is not None:
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(frame, channels="BGR", caption="Captured Image", use_column_width=True)
    
    # Preprocess for model
    img = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
    if input_dtype == np.uint8:
        img = img / input_scale + input_zero_point
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.expand_dims(img, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    if output_dtype == np.uint8:
        predictions = (predictions.astype(np.float32) - output_zero_point) * output_scale
    
    max_index = np.argmax(predictions)
    confidence = float(predictions[max_index])
    label = labels[max_index].replace("Background", "").strip()
    
    # Display result with color
    if confidence >= 0.9 and label != "":
        st.success(f"This is {label} ({round(confidence*100, 2)}%)")
        tts.say(f"This is {label}")
        tts.runAndWait()
    else:
        st.warning("Move note clearly to camera")
        tts.say("Move note clearly to camera")
        tts.runAndWait()
