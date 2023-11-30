import numpy as np
import random

X = np.load('preprocessed_images.npy')
y = np.load('labels.npy')

print(len(y))
print(y.shape)