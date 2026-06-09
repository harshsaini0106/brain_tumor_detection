from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("brain_tumor_cnn.keras")

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    file_bytes = np.frombuffer(
        file.read(),
        np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_GRAYSCALE
    )

    img = cv2.resize(img, (128,128))
    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]

    prediction = "Tumor" if prob > 0.5 else "No Tumor"

    return jsonify({
        "prediction": prediction,
        "probability": float(prob)
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)