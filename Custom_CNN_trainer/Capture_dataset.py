import cv2
import os

# Define the object classes
object_classes = ['Class_0', 'Class_1', 'Class_2', 'Class_3']

# Set up the camera or video capture
camera = cv2.VideoCapture(0)  # Adjust the parameter if using a different camera

# Function to capture images in a burst for a specific class without overwriting existing images
def capture_burst_images(class_name, num_images):
    # Create a directory if it doesn't exist
    if not os.path.exists(class_name):
        os.makedirs(class_name)
    
    print(f"Capturing {num_images} images in a burst for {class_name}...")

    existing_images = len(os.listdir(class_name))
    for i in range(existing_images, existing_images + num_images):
        ret, frame = camera.read()
        img_name = os.path.join(class_name, f"{class_name}_{i}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")

    print(f"Finished capturing {num_images} images for {class_name} in a burst.")
    cv2.destroyAllWindows()

# Choose the object class and number of images for a burst
chosen_class = int(input(f"Enter the object class (0 to {len(object_classes) - 1}): "))
num_images_to_capture = int(input("Enter the number of images to capture in a burst: "))

if chosen_class >= 0 and chosen_class < len(object_classes):
    capture_burst_images(object_classes[chosen_class], num_images_to_capture)
else:
    print("Invalid object class. Please choose a valid number.")

# Release the camera
camera.release()
