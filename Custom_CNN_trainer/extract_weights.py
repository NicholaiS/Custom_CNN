import tensorflow as tf
import numpy as np

# Load your trained model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (3, 3), activation='relu', input_shape=(40, 30, 1)),
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

# Load the trained weights into the model
model.load_weights('custom_model_weights.h5')

# Save the weights of the model to a text file with comma separators
with open('model_weights.txt', 'w') as f:
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            for i, w in enumerate(layer.get_weights()):
                f.write(f'Layer Name: {layer.name}\n')
                f.write(f'Weight {i + 1} - Shape: {w.shape}\n')
                flattened_weights = w.flatten()  # Flatten the weights
                weights_str = ', '.join(map(str, flattened_weights))
                f.write(weights_str + '\n\n')
