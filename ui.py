# Import necessary libraries
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load the trained model
model = keras.models.load_model('resnet50.h5')  # Replace with the path to your trained model

# Set a title and description for your app
st.title("Tuberculosis Detection App")
st.write("Upload a chest X-ray image for TB detection.")

# Upload an image
uploaded_image = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

# Check if an image is uploaded
if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Add a button to initiate TB detection
    if st.button("Detect TB"):
        # Preprocess the uploaded image
        image = Image.open(uploaded_image)
        image = image.resize((224, 224))  # Resize to match the input size of your model
        image = np.array(image)
        #image = image / 255.0  # Normalize the image (assuming your model expects values in [0, 1])
        def threshold_predictions(predictions, threshold=0.5):
            return (predictions >= threshold).astype(int)
        # Example usage:
        threshold = 0.5  # Adjust this threshold as needed
        predictions = model.predict(np.expand_dims(image, axis=0))# Replace with your input data
        binary_predictions = threshold_predictions(predictions, threshold)
        if binary_predictions[0][0] == 1:
            st.write("The patient does not have TB.")
        elif binary_predictions[0][1] == 1:
            st.write("The patient has a different disease")
        else:
            st.write("The patient has TB.")