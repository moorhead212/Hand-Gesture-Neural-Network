import cv2
import numpy as np
from keras.models import load_model

# Load the model from the training. 
model = load_model('gesture_recognition_model.h5')

# Defining the gestures. 
gesture_classes = ["one", "palm", "peace"]

# Define a function to perform real-time recognition
def recognize_gesture(frame):
    # Convert the frame to grayscale (to match the models training), and resize and normalize the frame.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.resize(frame_gray, (224, 224))
    frame_gray = frame_gray / 255.0
    frame_gray = np.expand_dims(frame_gray, axis=0)

    # Use the model to predict what gesture is being shown. 
    prediction = model.predict(frame_gray)
    predicted_class_index = np.argmax(prediction)
    max_confidence = prediction[0][predicted_class_index]

    # Set a confidence threshold to show if it is a trained gesture or not.
    min_confidence = 0.5

    # If the confidence is high enough then display that along with the confidence - if not then display no gesture with confidence.
    if max_confidence >= min_confidence:
        predicted_class_name = gesture_classes[predicted_class_index]
    else:
        predicted_class_name = "No gesture"
    return predicted_class_name, max_confidence

# Opens the webcam to capture the frames
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully - informs if it fails. 
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# If connection was successful, read images
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    # Flips frame so it looks natural
    frame = cv2.flip(frame, 1)

    # Recognize the gesture
    recognized_gesture, max_confidence = recognize_gesture(frame)

    # Show the gesture and the confidence on screen.
    text = f"{recognized_gesture} (Confidence: {max_confidence:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame 
    cv2.imshow('Gesture Recognition', frame)

    # If q is pressed - exit the loop in order to close out. 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Disconnect from webcam and close all windows out. 
cap.release()
cv2.destroyAllWindows()
