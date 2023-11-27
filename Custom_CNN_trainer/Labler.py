import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm  # Import tqdm for the progress bar

# Define paths to folders containing images for each class
class_0_path = 'Class_0\\'
class_1_path = 'Class_1\\'
class_2_path = 'Class_2\\'
class_3_path = 'Class_3\\'

def create_custom_dataset(class_0_path, class_1_path, class_2_path, class_3_path):
    images = []
    labels = []

    total_images = sum(len(files) for _, _, files in os.walk(class_0_path)) + \
                   sum(len(files) for _, _, files in os.walk(class_1_path)) + \
                   sum(len(files) for _, _, files in os.walk(class_2_path)) + \
                   sum(len(files) for _, _, files in os.walk(class_3_path))
    
    progress_bar = tqdm(total=total_images, desc='Creating Dataset', unit='image')

    # Load images and assign labels
    for class_idx, class_path in enumerate([class_0_path, class_1_path, class_2_path, class_3_path]):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)  # Read image
            image = cv2.resize(image, (40, 30))  # Resize image as needed

            # Convert image to grayscale
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image_gray)
            labels.append(class_idx)  # Assign label

            progress_bar.update(1)  # Update progress bar

    progress_bar.close()  # Close progress bar after completion

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Create a custom dataset
custom_images, custom_labels = create_custom_dataset(class_0_path, class_1_path, class_2_path, class_3_path)

# Save the custom dataset using pickle
with open('custom_dataset.pkl', 'wb') as f:
    pickle.dump((custom_images, custom_labels), f)
