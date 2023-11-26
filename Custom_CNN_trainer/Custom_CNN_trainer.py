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
train_images, val_images, train_labels, val_labels = train_test_split(custom_images, custom_labels, test_size=0.2, random_state=42)

# Reshape images to match the model's input shape for grayscale images
train_images = train_images.reshape(-1, 32, 32, 1)
val_images = val_images.reshape(-1, 32, 32, 1)

# 2. Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(1, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(1, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# 3. Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Train the model
history = model.fit(train_images, train_labels, epochs=250, validation_data=(val_images, val_labels), batch_size=32)

# 5. Evaluate the model
test_loss, test_acc = model.evaluate(val_images, val_labels)
print(f'Validation accuracy: {test_acc}')

# 6. Saving the weights (optional)
model.save_weights('custom_model_weights.h5')
