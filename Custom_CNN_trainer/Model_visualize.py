import tensorflow as tf
import matplotlib.pyplot as plt

# Define your model
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

# Plot the model
tf.keras.utils.plot_model(model, to_file='my_cnn_model.png', show_shapes=True, show_layer_names=True)

# Display the model visualization
image = plt.imread('my_cnn_model.png')
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.axis('off')  # Hide axes
plt.show()

