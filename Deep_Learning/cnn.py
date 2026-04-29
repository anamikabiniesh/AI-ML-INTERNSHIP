import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="CIFAR-10 CNN", page_icon="🚗", layout="centered")

st.title("CIFAR-10 Image Classifier")
st.caption("Model: 3x Conv2D + 2x Dense. Trained on CIFAR-10")

MODEL_PATH = "cnn_model.h5"
class_names = ['Airplane','Automobile','Bird','Cat','Deer',
               'Dog','Frog','Horse','Ship','Truck']

# 1. Load model with error handling
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"'{MODEL_PATH}' not found. Make sure you ran your training script first.")
        st.info(f"Current folder: {os.getcwd()}")
        st.stop()
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("This usually means TensorFlow version mismatch. Try: pip install tensorflow==2.15.0")
        st.stop()

model = load_model()
st.success("Model ready")

# 2. File uploader
uploaded_file = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 3. Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Upload", use_container_width=True)

    # 4. Preprocess exactly like training data
    img_resized = image.resize((32, 32)) # CIFAR-10 is 32x32
    img_array = np.array(img_resized) / 255.0 # Same normalization as X_train / 255.0
    img_batch = np.expand_dims(img_array, axis=0) # Add batch dim: (1, 32, 32, 3)

    with col2:
        st.image(img_resized, caption="Resized to 32x32", width=150)

    # 5. Predict
    try:
        preds = model.predict(img_batch, verbose=0)[0] # Get first item from batch
        pred_idx = np.argmax(preds)
        pred_class = class_names[pred_idx]
        confidence = preds[pred_idx] * 100

        st.subheader("Result")
        st.metric(label="Prediction", value=pred_class, delta=f"{confidence:.1f}% confidence")

        # 6. Show all class probabilities
        st.subheader("All Probabilities")
        prob_dict = {class_names[i]: float(preds[i]) for i in range(10)}
        st.bar_chart(prob_dict)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("Upload an image to classify. Try a photo of a car, plane, dog, etc.")
