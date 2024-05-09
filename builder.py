# By Landon Moceri on 5/8/2024
# Written with the help of GitHub Copilot

import cv2
import os
import time
import numpy as np
import tensorflow as tf

# New imports
# from tensorflow.python.keras.models import Sequential, load_model
# from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
#from keras_preprocessing.image import load_img, img_to_array

# Old imports
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.preprocessing.image import load_img, img_to_array

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, PredefinedSplit
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from skimage import color

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
        # Sampled from the Lego bricks using MS Paint
        self.color_dict = {
            "#770A00": "red",  
            "#9D3500": "orange", 
            "#9F7204": "yellow",  
            "#123029": "green", 
            "#051E60": "blue",  
            "#16133B": "purple",  
            "#915B6C": "pink",  
            "#61595C": "lb_gray",  
            "#2A272A": "db_gray",  
            "#050608": "black",  
            "#9D908A": "white",  
            "#29120D": "brown"  
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

        # Get the pixel in the center of the frame
        pixel = frame[int(frame.shape[0]/2), int(frame.shape[1]/2)]

        # Convert the pixel to 8-bit unsigned integer format
        pixel = np.uint8(pixel)

        # Convert the pixel to Lab color space
        pixel_lab = color.rgb2lab([[[pixel / 255]]]).flatten()
        color_dict_lab = {col: color.rgb2lab([[[np.array(hex_to_rgb(col)) / 255]]]).flatten() for col in self.color_dict.keys()}

        # Find the closest match in the dictionary
        closest_color = min(color_dict_lab.keys(), 
                            key=lambda color: np.linalg.norm(color_dict_lab[color] - pixel_lab))

        print("Closest match in the dictionary:", self.color_dict[closest_color])
        
        # Return the color's name as a string
        return self.color_dict[closest_color]
    
    def overlay_bricklink(self, frame, col):
        
        # Find the path to the corresponding image from its color
        path = ""
        if col == None:
            path = "data/bl_images/logo.png"
        else:
            path = f"data/bl_images/{col}.png"
        
        if not os.path.isfile(path):
            print("The image file does not exist.")
            print(f"Path: {path}")
            return
        
        # Read in the image
        overlay = cv2.imread(path)
        
        # Get the desired height
        height = frame.shape[0]
        height = int(height *1.5)
        
        # Resize the image to 720*720
        overlay = cv2.resize(overlay, (height, height), interpolation = cv2.INTER_AREA)
        
        # 1.5x the size of the frame
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5)
        
        # Create a new image with additional width for the box
        frame = np.hstack((frame, overlay))
        
        return frame
        
        

# Written with the help of GitHub Copilot
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


if __name__ == "__main__":
    
    # Create the LegoClassifier object
    classifier = LegoClassifier("data/final_train_imgs2/")
    
    # Create or load in model
    # Check if the lego_classifer file exists
    if not os.path.exists("models/lego_classifier.keras"):
        classifier.setup_model()

    else:
        classifier.load_pretrained_model("models/lego_classifier.keras")
    
    # Open video capture
    print("Opening video capture.")
    start_time = time.time()
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print(f"Video capture opened, time to open video capture: {time.time() - start_time}")
    
    # Loop until the end of the video
    while(video.isOpened()):
        # Read the frame
        frame = video.read()[1]
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("The frame is not valid.")
            continue
        
        # Get the predicted label
        label = classifier.predict_frame(frame.copy())
        col = None
        
        # If the prediction is 1, then overlay the icon at the top left corner
        if label == 1:
            frame = classifier.impose_icon(frame)
        
            # Find the color of the brick in the frame
            col = classifier.find_color(frame.copy())
        
        # Draw a positioning box in the center of the frame
        cv2.rectangle(frame, (int(frame.shape[1]/2 - 75), int(frame.shape[0]/2 - 75)), 
                      (int(frame.shape[1]/2 + 75), int(frame.shape[0]/2 + 75)), (0, 0, 0), 2)
        
        # Overlay the bricklink data from the color
        frame = classifier.overlay_bricklink(frame, col)
        
        # Display the frame
        cv2.imshow('Builder', frame)
        
        # Define q
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    
    # Shut down video stream
    video.release()
    cv2.destroyAllWindows()