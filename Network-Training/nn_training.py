import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping

# Setting labels
gesture_classes = ["one", "palm", "peace"]

# Loading my preprocessed images/labels with 5000 images in each class.
X = np.load('preprocessed_images.npy')
y = np.load('labels.npy')

# Separating the training and testing images - 60/40 split.
# Stratify to avoid pulling all test images from the same class.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42)

# Defining model - Sequential with 6 Conv2D Layers each with a max pooling 2x2
# Flatten and Dense to allow for training
# Dense 3 to sort the images into the respective gestures.
model = Sequential([
    Conv2D(16, 5, activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, 3),
    MaxPooling2D((2, 2)),
    Conv2D(64, 3),
    MaxPooling2D((2, 2)),
    Conv2D(64, 3),
    MaxPooling2D((2, 2)),
    Conv2D(64, 3),
    MaxPooling2D((2, 2)),
    Conv2D(64, 3),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile using adam optimizer (self-adjusting), loss by category, and metric printing accuracy and validation accuracy.
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=['accuracy'])

# Define early stopping callback in case we over fit in later epochs we can take the best weights. 
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# 15 Epochs allows for a good balance between time to train and not running forever
num_epochs = 15

# Setting the batch size and training/testing validation. 
history = model.fit(X_train, y_train, batch_size=64, epochs=num_epochs,
                    verbose=1, validation_data=(X_test, y_test))

# Printing the evaluations after the training completes.
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Saving the model for use in real time prediction. 
model.save('gesture_recognition_model.h5')

# Predict classes for test set
y_pred = model.predict(X_test)

# Get predicted classes from the probabilities using argmax
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Print classification report and confusion matrix
classification_rep = classification_report(
    y_test_labels, y_pred_classes, target_names=gesture_classes)
print(classification_rep)

# Printing confusion matrix to see how accurately each gesture is being predicted. 
confusion_mat = confusion_matrix(y_test_labels, y_pred_classes)
print("Confusion Matrix:")
print(confusion_mat)

print("Training is complete.")

# Plotting the training loss and validation loss.
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

# Plotting the training accuracy and validation accuracy.
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()
