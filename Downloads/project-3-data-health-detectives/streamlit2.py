import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Function to load a model
def load_keras_model(model_path):
    model = load_model(model_path)
    return model

# Load your pre-trained Keras models
vgg19_model = load_keras_model('vgg19_best_model.h5')
model_improvement = load_keras_model('model-improvement-43-0.97.h5')

# Function to preprocess the image
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
st.title('Image Classification App')

# Choose model
model_choice = st.radio("Choose a model to use:", ('VGG19 Best Model', 'Model Improvement'))

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("")
    st.write("Classifying...")
    
    # Preprocess the image to be compatible with the model
    # Adjust the target_size to match the input size of your model
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    # Run the image through the model
    if model_choice == 'VGG19 Best Model':
        prediction = vgg19_model.predict(processed_image)
    else:
        prediction = model_improvement.predict(processed_image)
    
    # Convert the prediction into a human-readable format
    # For example, you could apply a softmax function to convert output to probabilities,
    # and then use np.argmax to get the index of the most likely class label
    # Here's a placeholder for what that might look like:
    labels = ['Class1', 'Class2', 'Class3']  # Update these with your actual class labels
    predicted_class = labels[np.argmax(prediction)]
    
    st.write(f'Prediction: {predicted_class}')
