# By Landon Moceri on 5/8/2024
# Written with the help of GitHub Copilot

import cv2
import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

# TEST_MODE = "SPLIT"
TEST_MODE = "SEPARATE"


class LegoClassifier:
    def __init__(self, lego_dir, non_lego_dir, lego_test_dir, non_lego_test_dir):
        self.lego_dir = lego_dir 
        self.non_lego_dir = non_lego_dir
        self.lego_test_dir = lego_test_dir
        self.non_lego_test_dir = non_lego_test_dir
        
        self.model = Sequential()
        self.X_train, y_train = None, None
        self.X_test, y_test = None, None
        
    def shuffle_data(self, X, y):
        X, y = shuffle(X, y)
        return X, y

    def load_data(self):
        
        if TEST_MODE == "SPLIT":
            # Load Lego images
            lego_imgs = [img_to_array(load_img(os.path.join(self.lego_dir, img), target_size=(64, 64), color_mode='grayscale')) for img in os.listdir(self.lego_dir)]
            lego_labels = [1] * len(lego_imgs)

            # Load non-Lego images
            non_lego_imgs = [img_to_array(load_img(os.path.join(self.non_lego_dir, img), target_size=(64, 64), color_mode='grayscale')) for img in os.listdir(self.non_lego_dir)]
            non_lego_labels = [0] * len(non_lego_imgs)

            # Combine data and labels
            X = np.array(lego_imgs + non_lego_imgs)
            y = np.array(lego_labels + non_lego_labels)
            
            # Shuffle data
            X, y = self.shuffle_data(X, y)

            # Perform train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        else:
            # Load Lego images
            lego_imgs = [img_to_array(load_img(os.path.join(self.lego_dir, img), target_size=(64, 64), 
                                            color_mode='grayscale')) for img in os.listdir(self.lego_dir)]
            lego_labels = [1] * len(lego_imgs)

            # Load non-Lego images
            non_lego_imgs = [img_to_array(load_img(os.path.join(self.non_lego_dir, img), target_size=(64, 64), 
                                                color_mode='grayscale')) for img in os.listdir(self.non_lego_dir)]
            non_lego_labels = [0] * len(non_lego_imgs)

            # Combine data and labels and convert to numpy arrays
            self.X_train = np.array(lego_imgs + non_lego_imgs)
            self.y_train = np.array(lego_labels + non_lego_labels)
            
            # Load Lego test images
            lego_imgs = [img_to_array(load_img(os.path.join(self.lego_test_dir, img), target_size=(64, 64), 
                                            color_mode='grayscale')) for img in os.listdir(self.lego_test_dir)]
            lego_labels = [1] * len(lego_imgs)

            # Load non-Lego test images
            non_lego_imgs = [img_to_array(load_img(os.path.join(self.non_lego_test_dir, img), target_size=(64, 64), 
                                                color_mode='grayscale')) for img in os.listdir(self.non_lego_test_dir)]
            non_lego_labels = [0] * len(non_lego_imgs)

            # Combine data and labels and convert to numpy arrays
            self.X_test = np.array(lego_imgs + non_lego_imgs)
            self.y_test = np.array(lego_labels + non_lego_labels)
        

    def preprocess_data(self):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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

    # Think about activation functions, number of layers, number of epochs


    def create_model(n_layers):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        for i in range(n_layers):
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def tune_model(self):
        self.model = KerasClassifier(build_fn=self.create_model, n_layers = 1, epochs=10, batch_size=32)
        param_grid = {'n_layers': [1, 2, 3, 4, 5]}
        # Perform grid search
        grid = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=3)
        grid_result = grid.fit(self.X_train, self.y_train)

        # Set classifier.model to the best model found by GridSearchCV
        classifier.model = grid.best_estimator_.model
    
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        
        
    def load_pretrained_model(self, model_path):
        self.model = load_model(model_path)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Loss: {loss}, Accuracy: {accuracy}")
        
    def predict(self, img_path):
        # Load the image in grayscale mode, resize it to 64x64, and convert it to an array
        img = img_to_array(load_img(img_path, target_size=(64, 64), color_mode='grayscale'))

        # Expand the dimensions to match the input shape of the model
        img = np.expand_dims(img, axis=0)

        # Predict the class of the image
        prob = self.model.predict(img)
        label = [1 if prob >= 0.5 else 0 for p in prob[0]]
        return label
        

if __name__ == "__main__":
    
    classifier = LegoClassifier("data/positive_train/", "data/negative_train/", "data/positive_test/", "data/negative_test/")
    
    # Create or load in model
    # if not os.path.exists("models"):
        
    print("No model found. Training new model.")
    classifier.load_data()
    classifier.preprocess_data()
    
    classifier.tune_model()
    
    classifier.train_model()
    
    #os.mkdir("models")

    #classifier.model.model.save("models/lego_classifier.h5")
        
    # else:
    #     print("Model already exists. Skipping training.")
    #     classifier.load_pretrained_model("models/lego_classifier.h5")
        
    # Evaluate model
    classifier.evaluate_model()
    
    # Test prediction
    path = "data/positive_test/IMG_9082.jpg"
    print("Prediction for: ", path)
    print(classifier.predict(path))

    
        
        