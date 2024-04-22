# Prediction of Presence of Brain Tumor from MRI Scans

<p align="center"> <img width="286" alt="Screenshot 2024-04-22 at 9 51 26 AM" src="https://github.com/deumanma/project-3-data-health-detectives/assets/17521916/910c5eec-08b5-407e-b785-979264fc849f">

## Project 3 MSU AI Bootcamp
Team Members: Betsy Deuman	Jasmine Harper	Dr. Chadi Saad	Aaron Wood

### Dependencies
- Python 3.7+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- OpenCV
- TensorFlow
- Scikit-Learn
- Google Collab
  
### Installation
Clone this repository, and install the packages in the dependent packages. Use the MRI_BRAIN_Tumor_ThursdayFinal to import into Google Collab and do the Data exploration, cleanup, augmentation, and model testing and training on four different models. The BrainTumorChatbot.py file can be used with the model-improvement-28-0.98.h5 (the highest accuracy model from the testing set) to run the chatbot. 

## Brain Tumor Classification Using Deep Neural Networks
Brain tumor classification is a critical task in medical imaging, aiding in the diagnosis and treatment planning for patients. Deep Learning, particularly Convolutional Neural Networks (CNNs), has shown remarkable success in this area. Here, we explore two effective design strategies for constructing deep neural networks capable of classifying brain tumors into four distinct types.

## Custom CNN Architecture 
### Design Overview
The Custom CNN Architecture is tailored specifically for the dataset at hand, allowing for a flexible design that can be iteratively refined to meet the unique requirements of brain tumor classification.

**Layers**

* Input layer: Description: The entry point for input images, resized to a uniform dimension
(e.g., 224x224 pixels) for consistency.
* Convulutional layers: Purpose: Extract features from images using small kernel sizes (e.g., 3x3 or 5x5) with ReLU activation for introducing non-linearity.
* Pooling layers: Function: Employ max pooling to reduce the spatial dimensions and computational load, improving efficiency.
* Normalization layers: Stabilize learning by applying batch normalization following convolutional layers.
* Fully connected layers: Action -  Flatten the output from convolutional and pooling layers, using dense layers for classification. Dropout layers are included to mitigate overfitting.
* Output Layer: Goal - Classify images into one of the four brain tumor categories using a softmax activation function.

### Selecting The Right Approach

* Custom CNN Architecture: Best if you possess the computational resources and domain expertise necessary for developing and refining a specialized model.
* Transfer Learning with Pretrained Models: Ideal for scenarios with limited data or when aiming to leverage pre-established patterns from extensive trained models, offering faster training and enhanced accuracy.


Both strategies necessitate meticulous hyperparameter tuning (learning rate, batch size, number of epochs) and ongoing performance evaluation using a validation set to ensure the model's effectiveness and adjust as needed.

## Deep Neural Network Diagrams

Custom CNN Architecture: This diagram illustrates a custom Convolutional NeuralNetwork (CNN) specifically designed for the task of classifying brain tumors. It highlights the sequential layers including convolutional, max pooling, batch for normalization, and dense layers, culminating in an output layer with softmax activation for classification into four categories.
<p align="center"> <img width="619" alt="Screenshot 2024-04-22 at 11 14 09 AM" src="https://github.com/deumanma/project-3-data-health-detectives/assets/17521916/983a72b3-bae4-4a0e-8b61-b86985721337">

Transfer Learning Architecture: This diagram showcases a network utilizing transfer learning from a pre-trained base model (like VGG16, ResNet50, or InceptionV3), followed by a global average pooling layer, dense layers with dropout for classification, and a softmax output layer. This approach leverageslearned features from large datasets to improve classification accuracy.
<p align="center"> <img width="618" alt="Screenshot 2024-04-22 at 11 14 20 AM" src="https://github.com/deumanma/project-3-data-health-detectives/assets/17521916/c451abd9-ac0e-4c1e-87b6-311dfb394024">

## Set up a Transfer Learning Architecture with the Brain Tumor MRI Dataset

This guide outlines a comprehensive approach to developing a transfer learning model using the Brain Tumor MRI Dataset available on Kaggle. By leveraging a pretrained model, we can efficiently learn from image data to classify brain tumors with high accuracy.

### Prepare Your Environment
Before diving into the data and model, ensure your environment is correctly set up:

Required Software & Libraries:
Python: The core programming language we'll be using.
Libraries: Make sure to have tensorflow , keras , numpy , pandas , and matplotlib installed.
Dataset Acquisition:
Kaggle Dataset: The Brain Tumor MRI Dataset is available on Kaggle. You'll need a Kaggle account and the Kaggle API installed on your machine. Use the Kaggle CLI for direct dataset download.

### 🔄 2. Load and Preprocess the Data
Loading the Data
Image Libraries: Utilize PIL or opencv to load the images.
Dataset Segmentation: If the dataset isn't already divided, split it into training,
validation, and test sets.

Preprocessing Steps
Resize Images: Adjust the images to the expected size of the pretrained
model (e.g., 224x224 for VGG16).
Normalization: Scale pixel values to a 0-1 range for better model performance.
Data Augmentation: Enhance your model's robustness and reduce overfitting
with ImageDataGenerator from keras.preprocessing.image .

### 🧠 3. Configure the Pretrained Model
Model Selection: Choose from the pretrained models available in
tensorflow.keras.applications , such as VGG16, ResNet50, or InceptionV3.
Model Configuration: Load the chosen model without its top layer
( include_top=False ) and set input_shape to match your image size.
### ➕ 4. Add Classification Layers
Global Average Pooling (GAP): Integrate a GlobalAveragePooling2D layer to condense feature maps into a single vector per channel.
Dense Layers: Insert one or two dense layers to learn complex feature combinations, applying dropout or other regularization techniques to avoid overfitting.
Output Layer: The final layer should be a dense layer with a softmax activation
function, having a neuron for each brain tumor class.

### 🛠 5. Compile the Model
Configure your model for training:
Optimizer: Use an optimizer like Adam for effective learning.
Loss Function: Apply categorical_crossentropy for multi-class classification.
Metrics: Evaluate your model with metrics such as accuracy.

### 🏋 6. Train the Model
Train your model with the .fit() method, providing:
Training and Validation Data: Ensure both datasets are included.
Epochs & Batch Size: Set these based on your dataset size and computational
resources.
Callbacks: Use ModelCheckpoint and EarlyStopping to monitor performance and
save the best iteration.

### 🔍 7. Evaluate the Model
Performance Analysis: After training, use .evaluate() to test your model on the unseen test dataset.
Prediction: Employ .predict() to classify new images, further examining the model's predictive capabilities.

### 🔄 8. Fine-tuning (Optional)
For potential improvements:
Layer Unfreezing: Unfreeze and retrain some of the top layers of the pretrained model with a very low learning rate to refine feature learning on your specific dataset.
This step-by-step guide aims to streamline the development of a transfer learning  model for classifying brain tumors using MRI images. By following these instructions, you can harness the power of deep learning to contribute valuable insights into brain tumor diagnosis and classification.
