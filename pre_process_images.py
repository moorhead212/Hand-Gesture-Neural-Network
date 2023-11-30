import os
import glob
import json
import cv2
import numpy as np

# Setting the absolute path to the data and setting labels.
absolute_data_path = 'C:/NON_ONEDRIVE/480Project/data/bigsample'
gesture_classes = ["one", "palm", "peace"]

# Gathering 5000 images per class. 
N_IMAGES_PER_CLASS = 5000

# Initializing the lists for images and labels.
X = []
y = []

print("Starting Processing Gestures")

# For loop to go through each gesture and gather the set number of images for each gesture. 
# Annotations are stored in json format given by the dataset. 
for gesture in gesture_classes:
    absolute_annotation_path = f'C:/NON_ONEDRIVE/480Project/data/labels/{gesture}.json'
    with open(absolute_annotation_path, 'r') as json_file:
        result = json.load(json_file)
    
    # Gathers correct number of images for each class.
    path = os.path.join(os.getcwd(), absolute_data_path, gesture, '*.jpg')
    all_images = glob.glob(path)
    selected_images = all_images[:N_IMAGES_PER_CLASS]

    # For loop to convert image to grayscale, resize, and normalize each image. 
    for im_file in selected_images:
        img = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0

        # Getting the image name from the path and setting it as the key to search the json for 
        img_key = os.path.splitext(os.path.basename(im_file))[0]

        # If the image exists in the json - add the image to the list, and one-hot encode the labels. 
        if img_key in result:
            labels = result[img_key]['labels']
            if labels[0] == gesture:
                X.append(img)
                class_label = [0] * len(gesture_classes)
                class_label[gesture_classes.index(labels[0])] = 1
                y.append(class_label)

# Printing to verify everything selected properly.
    print(f"Processing gesture: {gesture} - Collected {len(selected_images)} images")

# Convert to a np array to save for use in training. 
X = np.array(X)
y = np.array(y)

# Save np array for training.
np.save('preprocessed_images.npy', X)
np.save('labels.npy', y)
