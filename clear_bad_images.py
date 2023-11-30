import os
from PIL import Image
import json

relative_data_path = 'C:/NON_ONEDRIVE/480Project/data/bigsample'
gesture_classes = ["one", "palm", "peace"]

TARGET_WIDTH = 1440
TARGET_HEIGHT = 1920

# Load label data if needed
label_data = {}  # Placeholder for label data

# Load label data from JSON files
for gesture in gesture_classes:
    relative_annotation_path = f'C:/NON_ONEDRIVE/480Project/data/labels/{gesture}.json'
    with open(relative_annotation_path, 'r') as json_file:
        label_data[gesture] = json.load(json_file)

for gesture in gesture_classes:
    gesture_folder = os.path.join(relative_data_path, gesture)
    image_files = [f for f in os.listdir(gesture_folder) if os.path.isfile(os.path.join(gesture_folder, f))]
    
    for image_file in image_files:
        image_path = os.path.join(gesture_folder, image_file)

        # Open the image using PIL
        img = Image.open(image_path)

        # Check image dimensions
        if img.size[0] != TARGET_WIDTH or img.size[1] != TARGET_HEIGHT:
            print(f"Deleting {image_file} due to incorrect dimensions.")
            os.remove(image_path)
            continue

        # Check labels
        img_key = os.path.splitext(image_file)[0]
        if img_key in label_data[gesture]:
            labels = label_data[gesture][img_key]['labels']
            
            # Perform label check here based on your specific label requirements
            # For example, check if the image matches the required label(s)
            # Modify this condition according to your label requirements
            if len(labels) != 1 or labels[0] != gesture:
                print(f"Deleting {image_file} due to incorrect label.")
                os.remove(image_path)
                continue

print("Processing completed.")
