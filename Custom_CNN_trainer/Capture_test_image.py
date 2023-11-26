import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_and_display_image(image_path):
    # Capture an image using cv2
    cap = cv2.VideoCapture(0)  # Change to 1 if using an external camera
    ret, frame = cap.read()
    cap.release()
    
    # Convert the captured image to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the grayscale image to 32x32 pixels
    resized_image = cv2.resize(gray_image, (32, 32))
    
    # Normalize pixel values to range between 0 and 1
    normalized_image = resized_image / 255.0
    
    # Display the image using Matplotlib in a larger size
    plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
    plt.imshow(normalized_image, cmap='gray')  # Change 'gray' to other colormaps if needed
    plt.axis('off')  # Hide axes
    plt.title('Resized Image')
    plt.show()
    
    # Flatten the normalized resized image
    flattened_image = normalized_image.flatten()
    
    # Save the flattened image as a single line in a text file with comma-separated values
    np.savetxt(image_path, flattened_image.reshape(1, -1), delimiter=',', fmt='%f')

    print(f"Image saved as {image_path}")

# Path to save the image as a text file
image_save_path = "resized_image.txt"

# Call the function to capture, preprocess, and display the image before saving it as a text file
preprocess_and_display_image(image_save_path)
