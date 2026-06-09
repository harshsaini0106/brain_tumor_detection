from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# =========================
# SAFE TF CONFIG
# =========================
tf.keras.backend.clear_session()
tf.get_logger().setLevel('ERROR')

print("Loading model...")
model = load_model("brain_tumor_cnn.keras")
print("Model loaded successfully")

# =========================
# MODEL WARMUP (CRITICAL FIX)
# =========================
dummy = np.zeros((1, 128, 128, 1), dtype=np.float32)
model.predict(dummy, verbose=0)
print("Model warmup done")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    print("Prediction request received")

    try:
        # =========================
        # CHECK INPUT FILE
        # =========================
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # =========================
        # SAFE IMAGE DECODING FIX
        # =========================
        file_bytes = np.frombuffer(file.read(), np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # FIX (not grayscale)

        if img is None:
            return jsonify({"error": "Invalid image file"}), 400

        # =========================
        # CONVERT TO GRAYSCALE SAFELY
        # =========================
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # =========================
        # PREPROCESS
        # =========================
        img = cv2.resize(img, (128, 128))
        img = img.astype("float32") / 255.0

        img = np.expand_dims(img, axis=-1)   # (128,128,1)
        img = np.expand_dims(img, axis=0)    # (1,128,128,1)

        print("Before prediction")

        # =========================
        # PREDICTION (SAFE)
        # =========================
        prob = model.predict(img, verbose=0)

        # FIX: avoid freezing float conversion issues
        prob = float(prob.squeeze())

        print("After prediction")

        # =========================
        # RESULT
        # =========================
        prediction = "Tumor" if prob > 0.5 else "No Tumor"

        print(f"Prediction: {prediction}, Probability: {prob}")

        return jsonify({
            "prediction": prediction,
            "probability": prob
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )