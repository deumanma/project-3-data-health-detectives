import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
# Function to load a model
def load_keras_model(model_path):
    return load_model(model_path)
# Assuming these are your model paths
vgg19_model = load_keras_model('/Users/aaronwood/desktop/project-3-data-health-detectives/vgg19_best_model.h5')
model_improvement = load_keras_model('/Users/aaronwood/desktop/project-3-data-health-detectives/model-improvement-02-0.87.h5')
# Preprocess the image to match the model's input requirements
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image
# Simulated ChatGPT analysis via API call (conceptual)
def get_chatgpt_analysis(prediction):
    api_url = "https://api.openai.com/v4/completions"
    headers = {
        "Authorization": f"Bsk-N65WRiAsK09FB7BxEYgJT3BlbkFJ6MFnSEvkHTMBWnjspbFp",
        "Content-Type": "application/json",
    }
    prompt = f"The MRI model predicted a {'tumor' if prediction > 0.5 else 'no tumor'}. Can you provide an analysis?"
    data = {
        "model": "text-davinci-003",  # Adjust based on the model you have access to
        "prompt": prompt,
        "temperature": 0.5,
        "max_tokens": 150,
    }
    response = requests.post(api_url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    else:
        return "Failed to get analysis from ChatGPT."
# Streamlit UI
st.title('MRI Tumor Detection and Analysis App')
model_choice = st.radio("Choose a pre-trained model for analysis:", ['VGG19 Model', 'Model Improvement'])
analysis_choice = st.radio("Choose the analysis method:", ['Direct Model Prediction', 'Detailed Analysis with ChatGPT'])
uploaded_file = st.file_uploader("Upload an MRI image:", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
    processed_image = preprocess_image(image)
    if model_choice == 'VGG19 Model':
        prediction = vgg19_model.predict(processed_image).flatten()[0]
    else:  # Model Improvement
        prediction = model_improvement.predict(processed_image).flatten()[0]
    if analysis_choice == 'Direct Model Prediction':
        predicted_class = 'Tumor Detected' if prediction > 0.5 else 'No Tumor Detected'
        st.write(f'Prediction: {predicted_class}')
    else:
        # Detailed analysis with ChatGPT
        chatgpt_response = get_chatgpt_analysis(prediction)
        st.write(chatgpt_response)
