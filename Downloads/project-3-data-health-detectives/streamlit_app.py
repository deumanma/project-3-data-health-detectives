import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the saved models
models = {
    "vgg19": load_model('vgg19_best_model.h5'),
    "model_improvement": load_model('model-improvement-34-0.94.h5')
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims / 255.

def predict_image_with_model(image_file, model_name):
    if model_name not in models:
        raise ValueError("Model not found")

    model = models[model_name]
    processed_image = preprocess_image(image_file)
    prediction = model.predict(processed_image)
    predicted_class = 'Yes' if prediction[0][0] > 0.5 else 'No'
    return predicted_class

# Streamlit UI
st.title('Image Prediction App')

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
model_name = st.selectbox(
    'Choose a model for prediction',
    ('vgg19', 'model_improvement')
)

if uploaded_file is not None and model_name:
    if allowed_file(uploaded_file.name):
        with st.spinner('Predicting...'):
            result = predict_image_with_model(uploaded_file, model_name)
            st.success(f'Prediction: {result} (using {model_name})')
    else:
        st.error('Unsupported file type. Please upload a PNG, JPG, or JPEG file.')
