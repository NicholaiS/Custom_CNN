import cv2
import numpy as np

def capture_and_save_image(image_path):
    # Capture an image using cv2
    cap = cv2.VideoCapture(0)  # Change to 1 if using an external camera
    ret, frame = cap.read()
    cap.release()

    # Convert the captured image to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the grayscale image to 40x30 pixels
    resized_image = cv2.resize(gray_image, (40, 30))
    
    # Normalize pixel values to range between 0 and 1
    normalized_image = resized_image / 255.0
    
    # Show the resized image using cv2
    normalized_image = cv2.resize(normalized_image, (640, 480))
    cv2.imshow('Resized Image (40x30)', normalized_image)
    cv2.waitKey(0)  # Wait until any key is pressed
    cv2.destroyAllWindows()  # Close the image window
    
    # Flatten the normalized resized image
    flattened_image = normalized_image.flatten()
    
    # Save the flattened image as a single line in a text file with comma-separated values
    np.savetxt(image_path, flattened_image.reshape(1, -1), delimiter=',', fmt='%f')
    print(f"Image saved as {image_path}")

# Path to save the image as a text file
image_save_path = "resized_image.txt"

# Call the function to capture the image and save it as a text file
capture_and_save_image(image_save_path)
