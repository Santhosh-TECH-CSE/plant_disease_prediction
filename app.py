import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


# -------------------------------
# Set Page Config
# -------------------------------
st.set_page_config(page_title="Plant Disease Prediction", layout="centered")


# -------------------------------
# Get Working Directory
# -------------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))


# -------------------------------
# Model Path
# -------------------------------
model_path = os.path.join(working_dir, "plant_disease_model.h5")




# -------------------------------
# Load Model (Cached for speed)
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()


# -------------------------------
# Load Class Indices
# -------------------------------
class_indices = json.load(open(os.path.join(working_dir, "class_indices.json")))


# -------------------------------
# Image Preprocessing
# -------------------------------
def load_and_preprocess_image(image, target_size=(224, 224)):

    img = image.resize(target_size)

    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array.astype("float32") / 255.0

    return img_array


# -------------------------------
# Prediction Function
# -------------------------------
def predict_image_class(model, image, class_indices):

    processed_img = load_and_preprocess_image(image)

    predictions = model.predict(processed_img)

    predicted_class_index = np.argmax(predictions)

    confidence = np.max(predictions)

    predicted_class_name = class_indices[str(predicted_class_index)]

    return predicted_class_name, confidence


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌿 Plant Disease Prediction System")

st.write("Upload a leaf image and the CNN model will predict the disease.")


uploaded_image = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_image is not None:

    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=250)

    with col2:

        if st.button("Predict Disease"):

            prediction, confidence = predict_image_class(
                model,
                image,
                class_indices
            )

            st.success(
                f"Prediction: {prediction}"
            )

            st.info(
                f"Confidence: {confidence*100:.2f}%"
            )