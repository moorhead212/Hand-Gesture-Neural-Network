# Hand-Gesture-Neural-Network
Created By: Kyle Moorhead

Created for COSC 480

Dataset: HAGrid (https://www.kaggle.com/datasets/kapitanov/hagrid)

## Training Information:
Dataset trained - HAGrid dataset, using "one", "palm", "peace" gestures.

# Info for Gesture Recognition

## To Run gesture detecion using trained model:
### Required installs for gesture detection:
tensorflow
cv2
numpy
keras.models

### Running detect_gestures_webcam.py:
1. Open "detect_gestures_webcam.py" (NOTE: Webcam required)
2. Run "detect_gestures_webcam.py"


# Info for Training  

## To run pre_process_images:
### Required installs/imports for pre_process_imgaes.py:
os
glob
json
cv2
numpy

### Prior to run pre_process_images:
1. Open clear_bad_images.py, change route to match your storage images location.
2. Run clear_bad_images.py (NOTE: Images are not all matching sizes and some have multiple labels, we delete these prior to pre_processing.)

### Running pre_process_images.py:
1. Open pre_process_images.py, change route to match your local files.
2. Update gesture_classes to gestures you're training.
3. Set N_IMAGES_PER_CLASS as you see fit. 5000 seemed to work well.

## To run nn_training.py:
### Required installs/imports for nn_training.py
matplotlib
numpy
sklearn
keras

### Running nn_trainin.py:
1. Open nn_training.py, change gestures_classes to match the gestures you are training.
2. Adjust Hyperparameters as you see fit. The current seemed to produce the best results. 
