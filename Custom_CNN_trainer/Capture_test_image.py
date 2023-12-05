import cv2
import numpy as np
import os

def capture_and_save_image(image_path, folder_paths, image_number):
    all_data = []  # List to store data from all classes
    folder_labels = []  # List to store folder labels
    
    for folder_path in folder_paths:
        # Get a list of image files in the folder
        image_files = os.listdir(folder_path)
        if len(image_files) <= image_number:
            print(f"Not enough images in {folder_path} to retrieve image number {image_number}")
            continue
        
        # Sort the image files to ensure consistent order
        image_files.sort()
        
        # Retrieve the specified image number (7500) from the folder
        image_file = os.path.join(folder_path, image_files[image_number - 1])  # Adjust index to start from 0
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
        
        if img is None:
            print(f"Failed to read {image_file}")
            continue
        
        # Resize the grayscale image to 40x30 pixels (if needed)
        resized_image = cv2.resize(img, (40, 30))
        
        # Show the resized image using cv2
        cv2.imshow('Resized Image (40x30)', resized_image)
        cv2.waitKey(0)  # Wait until any key is pressed
        cv2.destroyAllWindows()  # Close the image window
        
        # Flatten the resized image
        flattened_image = resized_image.flatten()
        
        # Append the flattened image data and folder label to the lists
        all_data.append(flattened_image)
        folder_labels.append(folder_path)
        print(f"Image data appended from folder: {folder_path}")
    
    # Convert the list of data arrays and labels to numpy arrays
    all_data = np.array(all_data)
    folder_labels = np.array(folder_labels)
    
    # Save the flattened image data and folder labels as a text file with integer values
    np.savetxt(image_path, np.column_stack((folder_labels, all_data)), delimiter=',', fmt='%s', header='Folder,Pixel_Values')
    print(f"All image data with folder labels saved in {image_path}")

# Path to save the image data as a text file
image_save_path = "resized_image.txt"

# Paths to the folders containing images for each class
folder_paths = ["Class_0", "Class_1", "Class_2", "Class_3"]

# Image number to retrieve from each folder (7500)
image_number = 5000

# Call the function to capture the 7500th image from each folder and save all data with folder labels in a single text file
capture_and_save_image(image_save_path, folder_paths, image_number)
