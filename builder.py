# By Landon Moceri on 5/8/2024
# Written with the help of GitHub Copilot

import cv2
import os
import time
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, PredefinedSplit
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans

class LegoClassifier:
    def __init__(self, image_dir):
        # Directory containing the images
        self.image_dir = image_dir 
        
        # Variables for the model
        self.model = None
        
        self.X_train, y_train = None, None
        self.X_test, y_test = None, None
        
        # Parameters for the images
        # Defines the size of images passed into the model
        target_dim = 128
        self.target_size = (target_dim, target_dim)

        # Defines the percent of data to load in for training
        # There are 200 images for each color, times 12 colors, = 2400 images for Lego
        # There are 2000 images for the empty plate and another 400 for the plate with a hand above it
        self.percent_used = 0.1
        
        # Switch to include light bluish gray, might help with the accuracy
        self.include_light_bluish_gray = True

        # Defines the color mode of the images passed into the model
        self.color_mode = 'rgb'
        
        # Define the number of channels based on the color mode
        if self.color_mode == 'grayscale':
            self.channels = 1
        elif self.color_mode == 'rgb':
            self.channels = 3
        elif self.color_mode == 'rgba':
            self.channels = 4
        
        # Icon for detected images
        self.detected_icon = cv2.imread("data/old_images/icon.png")
        self.detected_icon = cv2.resize(self.detected_icon, (self.detected_icon.shape[1] // 8, 
                                                             self.detected_icon.shape[0] // 8), interpolation = cv2.INTER_AREA)
        
        # KMeans object for color quantization
        self.kmeans = KMeans(n_clusters=2)
        
        # Color dictionary
        # Data taken from https://rebrickable.com/colors
        self.color_dict = {
            "#C91A09": "red",  
            "#FE8A18": "orange", 
            "#F2CD37": "yellow",  
            "#237841": "green", 
            "#0055BF": "blue",  
            "#81007B": "purple",  
            "#FC97AC": "pink",  
            "#A0A5A9": "light_bluish_gray",  
            "#6C6E68": "dark_bluish_gray",  
            "#05131D": "black",  
            "#FFFFFF": "white",  
            "#582A12": "brown"  
        }
        
    # Helper function to shuffle the data
    def shuffle_data(self, X, y):
        X, y = shuffle(X, y)
        return X, y

    # Written with the help of GitHub Copilot
    def load_data(self):
        lego_imgs = []
        lego_labels = []
        
        # Load in the Lego images from each subdirectory
        for subdir in os.listdir(self.image_dir):
            if subdir == "empty" or subdir == "hand":
                continue
            
            if not self.include_light_bluish_gray and subdir == "lb_gray":
                continue
            
            for img in os.listdir(os.path.join(self.image_dir, subdir)):
                img_path = os.path.join(self.image_dir, subdir, img)
                lego_imgs.append(img_to_array(load_img(img_path, target_size=self.target_size, color_mode=self.color_mode)))
                lego_labels.append(1)
           
        # Load in the non-Lego images from both subdirectories     
        non_lego_imgs = []
        non_lego_labels = []
        
        empty_dir = os.path.join(self.image_dir, "empty/")
        hand_dir = os.path.join(self.image_dir, "hand/")
        
        # Load in the non-Lego images, they are not in subdirectories
        for img in os.listdir(empty_dir):
            img_path = os.path.join(empty_dir, img)
            non_lego_imgs.append(img_to_array(load_img(img_path, target_size=self.target_size, color_mode=self.color_mode)))
            non_lego_labels.append(0)
            
        for img in os.listdir(hand_dir):
            img_path = os.path.join(hand_dir, img)
            non_lego_imgs.append(img_to_array(load_img(img_path, target_size=self.target_size, color_mode=self.color_mode)))
            non_lego_labels.append(0)
        
        # Combine Lego and non-Lego images
        X = np.array(lego_imgs + non_lego_imgs)
        y = np.array(lego_labels + non_lego_labels)
        
        # Shuffle the data
        X, y = self.shuffle_data(X, y)
        
        test_size = self.percent_used * 0.2
        train_size = self.percent_used * 0.8
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, 
                                                                                train_size=train_size, random_state=42)
        
        print("Data loaded successfully.")

    # Written with the help of GitHub Copilot
    def create_model(self, n_layers=1, epochs=5, batch_size=32):
        # Create the model
        model = Sequential()
        
        # Add the input layer, with the shape of the input images and the number of channels
        shape = self.target_size + (self.channels,)
        model.add(Input(shape=shape))
        
        # Add the first convolutional layer and pooling layer with 32 filters
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Add the remaining convolutional layers and pooling layers
        for i in range(n_layers):
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
        # Flatten the output of the convolutional layers 
        model.add(Flatten())
        
        # Add the dense layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
    
    def find_best_params(self, epochs = 5, batch_size = 32):
        print(f'Finding Best Parameters.')
        
       # Define the hyperparameters
        n_layers_options = [1, 2, 3]
        epochs_options = [5, 10, 15]
        batch_size_options = [32, 64, 128]

        # Initialize the variables to store the best parameters and the best score
        best_params = None
        best_score = 0

        # Iterate over all combinations of hyperparameters
        for n_layers in n_layers_options:
            for epochs in epochs_options:
                for batch_size in batch_size_options:
                    # Create and train the model with the current hyperparameters
                    model = self.create_model(n_layers=n_layers)
                    model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

                    # Evaluate the model on the validation data
                    score = model.evaluate(self.X_test, self.y_test)[1]

                    # If the score is better than the best score so far, update the best parameters and the best score
                    if score > best_score:
                        best_params = (n_layers, epochs, batch_size)
                        best_score = score
                        self.model = model

        # Print the best parameters and the best score
        print(f'Best parameters: {best_params}')
        print(f'Best score: {best_score}')
        
        return best_params
        
    def load_pretrained_model(self, model_path):
        # Print message
        print("Model already exists. Loading in from disc.")
        
        # Load the model
        classifier.model = tf.keras.models.load_model(model_path)
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Print message
        print("Model loaded successfully.")
        
    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Loss: {loss}, Accuracy: {accuracy}")
    
    def setup_model(self):
        # Print message
        print("No model found. Setting up new model")
        
        # Load the data
        self.load_data()
        
        # Find the best hyperparameters
        self.find_best_params()
        
        # Evaluate the model
        self.evaluate_model()
        
        # Create the directory
        os.makedirs("models", exist_ok=True)
        
        # Save the model
        tf.keras.models.save_model(self.model, 'models/lego_classifier.keras')
        print("Model saved successfully.")
        
    def predict_frame(self, frame):
        # Convert the frame to the color mode of the model
        if self.color_mode == 'grayscale':
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.color_mode == 'rgb':
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.color_mode == 'rgba':
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        else:
            print(f"Unexpected color mode: {self.color_mode}")
            return
        
        # Resize the image
        image = cv2.resize(image, self.target_size)
        
        # Expand the dimensions to match the input shape of the model
        image = np.expand_dims(image, axis=0)
        
        # Predict the class of the image
        prob = self.model.predict(image)
        
        # Print the prediction and return it
        if prob >= 0.5:
            print("LEGO")
            return 1
        else:
            print("none")
            return 0
        
    # Written with the help of GitHub Copilot
    def impose_icon(self, frame):
        # Overlay the icon at the top left corner
        frame[0:self.detected_icon.shape[0], 0:self.detected_icon.shape[1]] = self.detected_icon
        
        return frame
    
    # Written with the help of GitHub Copilot
    def find_color(self, frame):
        # Convert the frame to RGB
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        
        # Perform color quantization
        pixels = np.float32(frame.reshape(-1, 3))
        classifier.kmeans.fit(pixels)
        pixels = np.float32(classifier.kmeans.cluster_centers_[classifier.kmeans.labels_])
        quantized_image = pixels.reshape(frame.shape)

        # Convert the quantized image to 8-bit unsigned integer format
        quantized_image = np.uint8(quantized_image)

        # Count the occurrences of each color
        colors, counts = np.unique(quantized_image.reshape(-1, 3), axis=0, return_counts=True)

        # Find the second most common color
        second_most_common_color = colors[counts.argsort()[-2]]

        # Find the closest match in the dictionary
        def hex_to_rgb(hex_color):
            return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

        closest_color = min(self.color_dict.keys(), 
                            key=lambda color: np.linalg.norm(np.array(hex_to_rgb(color)) - second_most_common_color))

        print("Second most common color:", second_most_common_color)
        print("Closest match in the dictionary:", self.color_dict[closest_color])
        
        return self.color_dict[closest_color]

if __name__ == "__main__":
    
    classifier = LegoClassifier("data/final_train_imgs/")
    
    # Create or load in model
    # Check if the lego_classifer file exists
    if not os.path.exists("models/lego_classifier.keras"):
        classifier.setup_model()

    else:
        classifier.load_pretrained_model("models/lego_classifier.keras")
    
    # Open video capture
    print("Opening video capture.")
    start_time = time.time()
    video = cv2.VideoCapture(0)
    print(f"Video capture opened, time to open video capture: {time.time() - start_time}")
    
    # Loop until the end of the video
    while(video.isOpened()):
        # Read the frame
        frame = video.read()[1]
        
        # Get the predicted label
        label = classifier.predict_frame(frame.copy())
        
        # If the prediction is 1, then overlay the icon at the top left corner
        if label == 1:
            frame = classifier.impose_icon(frame)
        
        # Find the color of the brick in the frame
        classifier.find_color(frame.copy())
        
        # Display the frame
        cv2.imshow('Builder', frame)
        
        # Define q
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    
    # Shut down video stream
    video.release()
    cv2.destroyAllWindows()