# Prediction of Presence of Brain Tumor from MRI Scans
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
Clone this repository, and install the packages in the dependent packages. Use the MRI_BRAIN_Tumor_ThursdayFinal to import into Google Collab and do the Data exploration, cleanup, augmentation, and model testing and training on four different models. The BrainTumorChatbot.py file can be used with improvement

## Brain Tumor Classification Using Deep Neural Networks
Brain tumor classification is a critical task in medical imaging, aiding in the diagnosis and treatment planning for patients. Deep Learning, particularly Convolutional Neural Networks (CNNs), has shown remarkable success in this area. Here, we explore two effective design strategies for constructing deep neural networks capable of classifying brain tumors into four distinct types.

## Custom CNN Architecture 
### Design Overview
The Custom CNN Architecture is tailored specifically for the dataset at hand, allowing for a flexible design that can be iteratively refined to meet the unique requirements of brain tumor classification.

** Layers **

* Input layer: Description: The entry point for input images, resized to a uniform dimension
(e.g., 224x224 pixels) for consistency.
* Convulutional layers: Purpose: Extract features from images using small kernel sizes (e.g., 3x3 or 5x5) with ReLU activation for introducing non-linearity.
* Pooling layers: Function: Employ max pooling to reduce the spatial dimensions and computational load, improving efficiency.


