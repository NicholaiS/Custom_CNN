import tensorflow as tf
import numpy as np
import cv2

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

# Capture an image using OpenCV (you can also load an image using cv2.imread())
# Replace '0' with the appropriate camera index if using a webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture a frame
ret, frame = cap.read()

# Release the camera
cap.release()

# Check if the frame was captured successfully
if not ret:
    print("Error: Could not capture frame.")
    exit()

# Preprocess the frame for prediction (assuming frame is in grayscale)
processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
processed_frame = cv2.resize(processed_frame, (40, 30))  # Resize to match model input shape
processed_frame = np.expand_dims(processed_frame, axis=-1)  # Add a channel dimension

# Normalize pixel values
processed_frame = processed_frame / 255.0

# Display the captured image
cv2.imshow('Captured Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Prepare the image for prediction
image = processed_frame.reshape(1, 30, 40, 1)

# Get the prediction for the image
prediction = model.predict(image)
predicted_class = np.argmax(prediction)

# Print the predicted class
print(f"Predicted class: {predicted_class}")

# Wait for user input before exiting
input("Press Enter to exit...")
