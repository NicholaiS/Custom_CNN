import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to print layer inputs and outputs during model execution
class IntermediateOutputs(tf.keras.callbacks.Callback):
    def __init__(self, layer_index):
        super().__init__()
        self.layer_index = layer_index
        self.layer_input = None
        self.layer_output = None

    def on_predict_batch_end(self, batch, logs=None):
        intermediate_model = tf.keras.models.Model(inputs=self.model.input,
                                                   outputs=self.model.layers[self.layer_index].output)
        intermediate_output = intermediate_model.predict(self.layer_input)
        self.layer_output = intermediate_output

    def set_layer_input(self, input_data):
        self.layer_input = input_data

# Load the trained model architecture and weights
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
model.load_weights('custom_model_weights.h5')

# Create an instance of IntermediateOutputs for the third Conv2D layer (index 4)
intermediate_outputs_callback = IntermediateOutputs(layer_index=4)

# Capture an image using OpenCV (from webcam)
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
processed_frame = processed_frame / 255.0  # Normalize pixel values

# Display the captured image
cv2.imshow('Captured Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Prepare the image for prediction
image = processed_frame.reshape(1, 30, 40, 1)

# Set the input for the third Conv2D layer
intermediate_outputs_callback.set_layer_input(image)

# Get the output from the third Conv2D layer
model.predict(image, callbacks=[intermediate_outputs_callback])

# Display the output from the third Conv2D layer
print("Output from the third Conv2D layer:")
print(intermediate_outputs_callback.layer_output)

# Get the filter from the third Conv2D layer
layer_weights = model.layers[2].get_weights()[0]  # Extracting weights from the third Conv2D layer

# Display the filter matrix
print("Filter from the third Conv2D layer:")
print(layer_weights[:, :, 0, 0])

# Get the output from the last fully connected layer
output_fc = model.layers[-1].output  # Access the last layer
model_fc = tf.keras.models.Model(inputs=model.input, outputs=output_fc)

# Get the output from the fully connected layer
output_fc = model_fc.predict(image)
print("\nOutput from the fully connected layer (last array):")
print(output_fc)
