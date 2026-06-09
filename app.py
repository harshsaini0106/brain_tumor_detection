import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Brain Tumor AI",
    layout="centered"
)

@st.cache_resource
def load_my_model():
    return load_model("brain_tumor_cnn.h5")

model = load_my_model()

st.title(" Brain Tumor Detection")
st.markdown("---")
st.write("Upload an MRI scan")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Read image
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image = image.resize((128, 128))  # PIL resize (NO cv2)

    img = np.array(image)

    img_norm = img.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=(0, -1))

    # Show ONLY uploaded image
    st.subheader("Uploaded MRI")
    st.image(img, width=250, clamp=True)

    # Predict button
    if st.button("Predict Tumor"):
        try:
                with st.spinner("Analyzing MRI scan..."):

                    prob = float(model.predict(img_input, verbose=0).squeeze())
                    prediction = "Tumor" if prob > 0.5 else "No Tumor"

                

                if prediction == "Tumor":
                    st.error("Tumor Detected")
                else:
                    st.success("No Tumor Detected")

                st.subheader("Confidence Score")
                st.progress(prob if prediction == "Tumor" else 1 - prob)
                #st.write(f"Probability: **{prob:.2f}**")

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")