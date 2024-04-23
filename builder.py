# By Landon Moceri on 5/8/2024
# Written with the help of GitHub Copilot

import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator


class LegoClassifier:
    def __init__(self, lego_dir, non_lego_dir):
        self.lego_dir = lego_dir 
        self.non_lego_dir = non_lego_dir
        
        self.model = Sequential()
        self.X_train, y_train = None, None

    def load_data(self):
        # Load Lego images
        lego_imgs = [img_to_array(load_img(os.path.join(self.lego_dir, img), target_size=(64, 64))) for img in os.listdir(self.lego_dir)]
        lego_labels = [1] * len(lego_imgs)

        # Load non-Lego images
        non_lego_imgs = [img_to_array(load_img(os.path.join(self.non_lego_dir, img), target_size=(64, 64))) for img in os.listdir(self.non_lego_dir)]
        non_lego_labels = [0] * len(non_lego_imgs)

        # Combine data and labels and convert to numpy arrays
        X = np.array(lego_imgs + non_lego_imgs)
        y = np.array(lego_labels + non_lego_labels)

        return X, y

    def preprocess_data(self):
        datagen = ImageDataGenerator(
            rotation_range=20,     # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1,      # Randomly zoom image 
            width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,# randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images horizontally
            vertical_flip=True)   # you can also flip images vertically

        # fit parameters from data
        datagen.fit(self.X_train)

    def train_model(self, epochs = 10, batch_size = 32):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def create_model(self):
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        
        #model.add(Dense(num_classes, activation='softmax'))  # for multi-class classification
        # or
        self.model.add(Dense(1, activation='sigmoid'))  # for binary classification
        
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # for multi-class classification
        # or
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # for binary classification
    

if __name__ == "__main__":
    classifier = LegoClassifier("data/positive_train", "data/negative_train")
    classifier.create_model()
    # classifier.load_data()
    # classifier.preprocess_data()
    # classifier.train_model()