import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import cv2

# Load your custom dataset
with open('custom_dataset.pkl', 'rb') as f:
    custom_images, custom_labels = pickle.load(f)

# Normalize pixel values
custom_images = custom_images / 255.0

# Split dataset into train/validation
train_images, val_images, train_labels, val_labels = train_test_split(
    custom_images, custom_labels, test_size=0.2, random_state=42)

# Load the trained model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (3, 3), activation='relu', input_shape=(30, 40, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(1, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(1, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the trained weights
model.load_weights('custom_model_weights.h5')

# Choose the number of images for testing
num_test_images = 1000  # Replace with the desired number of test images

# Select a subset of validation images for testing
test_subset_images = val_images[:num_test_images]
test_subset_labels = val_labels[:num_test_images]

# Predict on the test subset
predictions = model.predict(test_subset_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy for each class
num_classes = 4  # Assuming 4 classes
for class_label in range(num_classes):
    class_indices = np.where(test_subset_labels == class_label)[0]
    class_predictions = predicted_labels[class_indices]
    class_accuracy = np.mean(class_predictions == class_label)
    print(f"Accuracy for Class {class_label}: {class_accuracy * 100:.2f}%")
