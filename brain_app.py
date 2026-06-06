import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("brain_tumor_cnn.keras")

st.title("Brain Tumor Detection")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_GRAYSCALE
    )
    #python -m streamlit run brain_app.py
    st.image(img)

    img = cv2.resize(img, (128,128))

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]

    #st.write(f"Tumor Probability: {prob:.4f}")

    if prob > 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")