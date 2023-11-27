import tensorflow as tf
import numpy as np
import pickle
import cv2
import random

# Load your custom dataset
with open('custom_dataset.pkl', 'rb') as f:
    custom_images, custom_labels = pickle.load(f)

# Normalize pixel values
custom_images = custom_images / 255.0

print(custom_images)

# Load the trained model architecture
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

# Load the trained model weights
model.load_weights('custom_model_weights.h5')

# Get 10 random indices for displaying random images
random_indices = random.sample(range(len(custom_images)), 10)

# Display 10 randomly selected images, their labels, and predictions
for idx in random_indices:
    # Display the image using OpenCV with an increased size
    img = cv2.resize(custom_images[idx], (600, 600))
    cv2.imshow(f'Image {idx} - Label: {custom_labels[idx]}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Prepare the image for prediction
    image = custom_images[idx].reshape(1, 30, 40, 1)

    # Get the prediction for the image
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Print the predicted class
    print(f"Predicted class for image {idx}: {predicted_class}, with the lable stating: {custom_labels[idx]}")

# Wait for user input before exiting
input("Press Enter to exit...")
