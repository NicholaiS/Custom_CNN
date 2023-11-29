import numpy as np
import tensorflow as tf

# Define the model architecture
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
    tf.keras.layers.Dense(4)  # Remove the softmax activation here
])

# Load the trained weights into the model
model.load_weights('custom_model_weights.h5')

# Read the contents from 'resized_image.txt'
with open('resized_image.txt', 'r') as file:
    content = file.read().split(',')

# Convert the content to a numpy array
flattened_matrix = np.array(list(map(float, content)))
flattened_matrix = flattened_matrix.reshape(1, 30, 40, 1)  # Reshape to match the input shape of the model

# Function to get intermediate layer outputs and shapes
layer_outputs = [layer.output for layer in model.layers]
layer_shapes = [layer.output_shape for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

# Perform inference to get values and shapes for each layer
activations = activation_model.predict(flattened_matrix)

# Write the layer values and shapes to a text file
with open('layer_values_with_dimensions.txt', 'w') as file:
    for i, (layer_activation, layer_shape) in enumerate(zip(activations, layer_shapes)):
        file.write(f"Layer {i}: {model.layers[i].name}\n")
        file.write(f"Shape: {layer_shape}\n")

        # Save the values with their respective dimensions
        if isinstance(layer_activation, np.ndarray):
            # Save the array with dimensions intact
            np.savetxt(file, layer_activation.squeeze(), fmt='%.8f')
        else:
            file.write(str(layer_activation))

        file.write('\n\n')

# Get and display the weights for layer 7
layer_7_weights = model.layers[7].get_weights()[0]  # Accessing weights
layer_7_bias = model.layers[7].get_weights()[1]  # Accessing biases

# Display the weights and biases for layer 7
print("Weights for Layer 7:")
print(layer_7_weights)
print("\nBiases for Layer 7:")
print(layer_7_bias)