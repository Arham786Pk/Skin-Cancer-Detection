📌 Skin Cancer Detection Using Deep Learning – Code Explanation
This project trains a Convolutional Neural Network (CNN) model to classify skin images as cancerous or non-cancerous using TensorFlow and Keras. The model is trained on labeled datasets and later used to make predictions on new skin lesion images.

📌 Step-by-Step Breakdown of the Code
1️⃣ Importing Required Libraries
NumPy & Pandas → For numerical and data manipulation.
Matplotlib → To plot training progress.
TensorFlow & Keras → To build and train the CNN model.
2️⃣ Data Preprocessing Using ImageDataGenerator
📌 What is ImageDataGenerator?
It is used to augment and preprocess images for better model generalization.

Train Data Generator

Rescales pixel values to [0,1] (normalization).
Applies random transformations (shear, zoom, horizontal flip) to create variations.
Test Data Generator

Only rescales images without augmentation.
✅ Why do we rescale images?
Deep learning models perform better when pixel values are in a small range (0 to 1) instead of (0 to 255).

3️⃣ Loading Training and Testing Data
The dataset is stored in train/test directories (path should be updated based on actual location).
Images are resized to 150x150 pixels and loaded in batches of 32.
The labels are binary (Cancer vs. No Cancer).
4️⃣ Building the CNN Model
The CNN architecture consists of:

Layer	Purpose
Conv2D (32 filters, 3x3) + ReLU	Extracts image features (edges, textures, etc.)
MaxPooling2D (2x2)	Reduces image size to prevent overfitting
Conv2D (64 filters, 3x3) + ReLU	Extracts deeper image features
MaxPooling2D (2x2)	Further reduces image size
Conv2D (128 filters, 3x3) + ReLU	Extracts complex patterns
MaxPooling2D (2x2)	Further reduces image size
Flatten()	Converts the 2D image matrix into a 1D vector
Dense (512, ReLU)	Fully connected layer for pattern recognition
Dense (1, Sigmoid)	Output layer (1 neuron) → predicts Cancer (1) or No Cancer (0)
✅ Why use ReLU activation?
It introduces non-linearity, making the model capable of learning complex patterns.

✅ Why use Sigmoid activation in the last layer?
Since it's a binary classification (Cancer vs. No Cancer), sigmoid outputs a probability between 0 and 1.

✅ Why use Binary Crossentropy as the Loss Function?
It's best for binary classification problems, ensuring the model learns the correct probabilities.

5️⃣ Training the Model
100 steps per epoch (based on dataset size).
10 epochs (model learns from the data 10 times).
Uses Adam optimizer (efficient for CNNs).
Tracks validation accuracy/loss to measure performance.
6️⃣ Evaluating the Model on Test Data
The model is evaluated using test data (not seen during training).
Accuracy is printed as final performance measure.
7️⃣ Plotting Training Results
Training vs Validation Accuracy Graph → Shows how well the model is learning.
Training vs Validation Loss Graph → Helps detect overfitting.
✅ Why do we check these graphs?
To ensure the model is not overfitting or underfitting.

8️⃣ Saving the Model for Future Use
The trained model is saved as a .h5 file so it can be reloaded later without retraining.

9️⃣ Making Predictions on New Images
The saved model is loaded for inference.
A new skin image (cancerous or non-cancerous) is loaded and resized to 150x150 pixels.
The model predicts:
If prediction < 0.5 → "No Cancer"
If prediction ≥ 0.5 → "Cancer Detected"
✅ Why convert the image into an array?
Deep learning models require numerical representations of images for processing.

✅ Why use expand_dims(img_array, axis=0)?
The model expects input in batch format, so we add an extra dimension.

📌 Summary of Key Concepts Used
🔹 Convolutional Neural Networks (CNNs) → Used for image classification.
🔹 ImageDataGenerator → Augments and pre-processes images.
🔹 Binary Classification (Cancer vs. No Cancer) → Uses Sigmoid activation.
🔹 Loss Function: Binary Crossentropy → Best for two-class problems.
🔹 Optimizer: Adam → Efficient weight updates for CNNs.
🔹 Saving & Loading Models → Avoids retraining from scratch.

📌 Expected Output
✅ Training Accuracy & Loss Graphs
✅ Final Model Accuracy on Test Data
✅ New Image Prediction ("Cancer Detected" or "No Cancer")

This project can be extended for multi-class classification (different skin disease types) and real-time mobile applications for skin cancer detection! 🚀
