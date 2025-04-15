"""
Semantic Segmentation Evaluation Script using Streamlit and TensorFlow

This script is used to load a pre-trained U-Net model, apply it to a dataset of images, 
and evaluate the model's performance by computing confusion matrices for each image. 
The results are saved as a NumPy array and the progress is displayed using Streamlit.

### Steps to Use the Script:

1. **Install Required Libraries**:
   Ensure that you have the required Python packages installed.
   You can install them via pip:

   pip install streamlit pandas numpy tensorflow scikit-learn


2. **Prepare Model and Dataset**:
- **Model**: Ensure you have a trained Keras U-Net model saved as a `.keras` file. 
  Update the `model_filepath` variable with the path to your saved model.
- **Dataset**: Modify or replace the `getPaths()` function in `utils.py` 
  to provide paths to your test images and their corresponding labels.

3. **Update Paths**:
Set the correct paths to your model and dataset in the script.

4. **Run the Script**:
You can run the script using Streamlit by running the following command:

    streamlit run script_name.py


5. **Output**:
- As the script runs, it will display a progress bar and compute confusion matrices 
  for each image. These matrices will be saved to a `benchmark.npy` file alternatively, you can modify the names of the files as well.

### Required Imports:
- `streamlit` for displaying progress and results.
- `tensorflow` for running the U-Net model.
- `sklearn.metrics.confusion_matrix` for evaluating model predictions.

### File Output:
- The confusion matrices for all the predictions will be saved as a NumPy file.
"""

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import parse_sample, convert_rgb_encoding_to_segmentation_map, rgb_to_class_id, getPaths, normalize

from sklearn.metrics import confusion_matrix

# 1. Specify the file path of the saved Keras model (replace with your model's path).
model_filepath = "C:/Users/talha/Projects/carpp/Checkpoint_simp/deep_more.checkpoint.model_v5.keras"

# 2. Get the paths of the images and labels using a custom function `getPaths()`.
x, y = getPaths()

# 3. Define a function to load the pre-trained model using the file path provided above.
#    This function is cached by Streamlit to avoid loading the model repeatedly.
@st.cache_resource
def load_model(checkpoint):
 """Load the pre-trained Keras model from the specified checkpoint file."""
 loaded_model = tf.keras.models.load_model(checkpoint)
 return loaded_model

# 4. Load the model.
loaded_model = load_model(model_filepath)

# 5. Initialize an empty list to store confusion matrices for each image's predictions.
confusion_matrices = []

# 6. Create a progress bar in the Streamlit app to track processing progress.
bar = st.progress(0, "Progress")

# 7. Loop over the dataset to process each image-label pair.
#    The loop applies model predictions to each image and calculates the confusion matrix.
for i in range(len(x)):
 # Preprocess the image and label using custom functions.
 image, label = parse_sample(x[i], y[i])
 image, label = normalize(image, label)

 # Add a batch dimension to the image to make it compatible with the model's input shape.
 image = tf.expand_dims(image, axis=0)

 # Perform model inference to predict the segmentation probabilities.
 probabilities = loaded_model.predict(image)

 # Convert the predicted probabilities into class predictions by taking the argmax.
 prediction = tf.argmax(probabilities, axis=-1)
 
 # Remove the batch dimension from the prediction and label.
 prediction = tf.squeeze(prediction)
 label = np.squeeze(label)
 
 # Flatten the prediction and label for evaluation purposes.
 prediction = prediction.numpy()
 pred = prediction.reshape((prediction.shape[0] * prediction.shape[1], ))
 actual = label.reshape((label.shape[0] * label.shape[1], ))
 
 # Compute the confusion matrix for the current prediction and label.
 cm = confusion_matrix(actual, pred, labels=range(0, 32))
 confusion_matrices.append(cm)
 
 # Update progress bar
 prog = int((i*100)/len(x))
 bar.progress(prog+1, "Progress")

# 8. Save the confusion matrices as a NumPy file.
np.save('benchmark.npy', np.array(confusion_matrices, dtype=object), allow_pickle=True)

# 9. Notify the user that the process is complete.
st.write("Done")
