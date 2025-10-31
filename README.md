# Currency Note Identifier

## About

This project is a simple currency note recognition system. It uses a trained TensorFlow Lite model to identify which currency note is shown in an input image.

## Contents

* `app.py` — Python script to run inference using the trained model.
* `labels.txt` — List of class labels corresponding to the model’s output.
* `model_quant.tflite` — Quantized TensorFlow Lite model file.
* `predictions.log` — Sample log of prediction outputs.
* `requirements.txt` — Python dependencies required.
* `runtime.txt` — Runtime specification (e.g., for deployment).
* `templates/` — Folder for any web UI templates (if applicable).

## Getting Started

1. Ensure you have Python installed (preferably 3.6+).
2. Clone the repository:

   ```bash
   git clone https://github.com/giri521/Currency-note-identifier.git
   cd Currency-note-identifier
   ```
3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```
4. Prepare an image of a currency note and run `app.py` pointing to the image to get the model’s predicted label.

## Usage

* Open `app.py` and check the image input section.
* The script loads `model_quant.tflite` and uses `labels.txt` to map the output indices to human-readable labels.
* The prediction result is printed (or logged) to `predictions.log`.

## Model & Data

* The model is a quantized TFLite version (`model_quant.tflite`) for efficient inference.
* `labels.txt` holds the classes the model was trained on (e.g., different denominations or currency types).
* No training code is included in the repository; only inference artifacts are provided.

## Notes & Limitations

* This version may support only certain denominations/currencies depending on the labels and training data.
* Accuracy depends heavily on image quality, lighting, orientation, and whether the note is clear in the image.
* For better results, use clean, well-lit images of the note, preferably aligned and fully visible.

* No explicit license file is present (check repository for updates).
* Contributions (bug fixes, model improvements, adding more denominations) are welcome — consider forking the repo and issuing a pull request.
