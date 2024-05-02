# By Landon Moceri on 5/8/2024
# Written with the help of GitHub Copilot

import cv2
import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, PredefinedSplit
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV


class LegoClassifier:
    def __init__(self, lego_model_dir, lego_image_dir, non_lego_dir):
        self.lego_model_dir = lego_model_dir 
        self.lego_image_dir = lego_image_dir
        self.non_lego_dir = non_lego_dir
        
        self.model = None
        self.X_train, y_train = None, None
        self.X_test, y_test = None, None
        
    def shuffle_data(self, X, y):
        X, y = shuffle(X, y)
        return X, y

    def load_data(self):
        
        # Load Lego model images
        lego_model_imgs = [img_to_array(load_img(os.path.join(self.lego_model_dir, img), target_size=(64, 64), color_mode='grayscale')) 
                           for img in os.listdir(self.lego_model_dir)]
        lego_model_labels = [1] * len(lego_model_imgs)
        
        # Load Lego images
        lego_real_imgs = [img_to_array(load_img(os.path.join(self.lego_image_dir, img), target_size=(64, 64), color_mode='grayscale')) 
                          for img in os.listdir(self.lego_image_dir)]
        
        # Augment lego real images
        lego_real_imgs = self.preprocess_data(np.array(lego_real_imgs))
        lego_real_labels = [1] * len(lego_real_imgs)
        
        # Put all Lego images together
        lego_imgs = lego_model_imgs + list(lego_real_imgs)
        lego_labels = lego_model_labels + lego_real_labels
        

        # Load non-Lego images
        non_lego_imgs = [img_to_array(load_img(os.path.join(self.non_lego_dir, img), target_size=(64, 64), color_mode='grayscale')) 
                         for img in os.listdir(self.non_lego_dir)]
        non_lego_labels = [0] * len(non_lego_imgs)
        
        
        # Combine Lego and non-Lego images
        X = np.array(lego_imgs + non_lego_imgs)
        y = np.array(lego_labels + non_lego_labels)
        
        # Shuffle the data
        X, y = self.shuffle_data(X, y)
        
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        

    def preprocess_data(self, X_train):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,     # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1,      # Randomly zoom image 
            width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,# randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images horizontally
            vertical_flip=True)   # you can also flip images vertically

        # Fit parameters from data
        datagen.fit(X_train)
        
        # Create a Python generator that yields augmented images
        generator = datagen.flow(X_train, batch_size=X_train.shape[0], shuffle=False)

        # Generate and return the augmented images
        X_train_augmented = next(generator)

        return X_train_augmented

    def train_model(self, epochs = 5, batch_size = 32):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    # Think about activation functions


    def create_model_helper(self, n_layers):
        model = Sequential()
        model.add(Input(shape=(64, 64, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        for i in range(n_layers):
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model    
    
    def create_model(self, epochs = 5, batch_size = 32):
        def model_fn(n_layers=1):
            return self.create_model_helper(n_layers)
        
        self.model = KerasClassifier(model=model_fn, n_layers = 1, epochs=epochs, batch_size=batch_size)
        param_grid = {'n_layers': [1, 2, 3, 4, 5]}
        
        # Perform grid search
        grid = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=3, verbose=2)
        grid_result = grid.fit(self.X_train, self.y_train)

        # Set classifier.model to the best model found by GridSearchCV
        self.model = grid.best_estimator_.model
    
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        
        # Get the best n_layers value and create the model with it
        best_n_layers = grid_result.best_params_['n_layers']
        self.model = self.create_model_helper(best_n_layers)
        
        
    def load_pretrained_model(self, model_path):
        # Print message
        print("Model already exists. Loading in from disc.")
        
        # Load the model
        classifier.model = tf.keras.models.load_model(model_path)
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Loss: {loss}, Accuracy: {accuracy}")
        
    def predict_image(self, img_path):
        # Load the image in grayscale mode, resize it to 64x64, and convert it to an array
        img = img_to_array(load_img(img_path, target_size=(64, 64), color_mode='grayscale'))

        # Expand the dimensions to match the input shape of the model
        img = np.expand_dims(img, axis=0)

        # Predict the class of the image
        prob = self.model.predict(img)
        label = [1 if prob >= 0.5 else 0 for p in prob[0]]
        
        print("Preditciton for " + img_path + ": " + str(label))
        return label
    
    def predict_image_array(self, dir_path):
        # Load the images from the directory in grayscale mode, resize it to 64x64, and convert it to an array
        imgs = [img_to_array(load_img(os.path.join(dir_path, img), target_size=(64, 64), color_mode='grayscale'))
                for img in os.listdir(dir_path)]
        
        # Expand the dimensions to match the input shape of the model
        imgs = np.array(imgs)
        imgs = np.expand_dims(imgs, axis=3)
        
        # Predict the class of the images
        probs = self.model.predict(imgs)
        labels = [1 if prob >= 0.5 else 0 for prob in probs]
        return labels
    
    def setup_model(self):
        # Print message
        print("No model found. Setting up new model")
        
        # Load the data
        self.load_data()
        
        # Create the model
        self.create_model()
        
        # Train the model
        self.train_model()
        
        # Evaluate the model
        self.evaluate_model()
        
        # Create the directory
        os.makedirs("models", exist_ok=True)
        
        # Save the model
        tf.keras.models.save_model(self.model, 'models/lego_classifier.keras')
        

if __name__ == "__main__":
    
    classifier = LegoClassifier("data/3d_model_lego_images/", "data/physical_lego_images/", "data/non_lego_images/")
    
    # Create or load in model
    # Check if the lego_classifer file exists
    if not os.path.exists("models/lego_classifier.keras"):
        classifier.setup_model()

    else:
        classifier.load_pretrained_model("models/lego_classifier.keras")
        
    
    # Test prediction
    # path = "data/old_images/plswork.jpg"
    # print(classifier.predict_image(path))
    
    # Load in the icon for detected images
    icon = cv2.imread("data/old_images/detected_icon.jpg")
    
    # Open video capture
    video = cv2.VideoCapture(1)
    
    # Loop until the end of the video
    while(video.isOpened()):
        # Read the frame
        frame = video.read()[1]
        
        original_image = frame
        
        # Convert the frame to grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert the frame to grayscale
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print(f"Unexpected number of channels in frame: {frame.shape}")
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize the image to 64x64
        image = cv2.resize(image, (64, 64))
        
        # Expand the dimensions to match the input shape of the model
        image = np.expand_dims(image, axis=0)
        
        # Predict the class of the image
        prob = classifier.model.predict(image)
        
        # Get the label
        label = [1 if prob >= 0.5 else 0 for p in prob[0]]
        
        # If the prediction is 1, then overlay the icon at the top left corner
        if label == 1:
            print("Detected Lego in loop")
            original_image[0:icon.shape[0], 0:icon.shape[1]] = icon
            
        print(label)
            
        
        # # Convert the image to color
        # # Check the number of dimensions and channels in the image
        # if len(image.shape) == 3 and image.shape[0] == 1:
        #     # Remove the extra dimension
        #     image = np.squeeze(image, axis=0)
        #     # Convert the image to BGR
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # else:
        #     print(f"Unexpected number of channels in image: {image.shape}")
        #     break
        
        # image = cv2.resize(image, (800, 600))
        
        # Display the frame
        cv2.imshow('Builder', original_image)
        
        # Define q
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    
    # Shut down video stream
    video.release()
    cv2.destroyAllWindows()

    
        
        