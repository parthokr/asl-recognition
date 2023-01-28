import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

data_dir = 'data'
classes = ['A', 'B', 'C']
X_train = []
y_train = np.array([])
for label in classes:
    y_train = np.append(y_train, [classes.index(label)]*len(os.listdir(f'{data_dir}/{label}')))

# print(y_train.shape)
for label in classes:
    for img in os.listdir(f'{data_dir}/{label}'):
        img = cv2.imread(f'{data_dir}/{label}/{img}')
        # X_train = np.append(X_train, [img])
        X_train.append(img)

X_train = np.array(X_train)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# print(X_train)
# print(y_test)

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(500, 500, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10)