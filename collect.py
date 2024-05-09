# Testing code written by Github Copilot
import cv2
import os

# Run the program multiple times

# Ask for a color name
color = input("Enter a color name (or 'q' to quit): ")

# Break the loop if the user enters 'q'
if color.lower() == 'q':
    exit()

# Open the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Create a directory for the color
os.makedirs(f'data/final_train_imgs2/{color}', exist_ok=True)

# Initialize frame count
count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        break

    # Save the resulting frame
    cv2.imwrite(f'data/final_train_imgs2/{color}/{color}_{count}.jpg', frame)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Increment frame count
    count += 1

    # Break the loop after capturing 200 frames
    if count >= 400:
        break

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# After the loop release the cap object
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()