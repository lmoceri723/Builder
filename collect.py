import cv2

# Run the program multiple times
while True:
    # Ask for a color name
    color = input("Enter a color name (or 'q' to quit): ")

    # Break the loop if the user enters 'q'
    if color.lower() == 'q':
        break

    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Initialize frame count
    count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            break

        # Save the resulting frame
        cv2.imwrite(f'data/final_train_imgs/{color}_{count}.jpg', frame)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Increment frame count
        count += 1

        # Break the loop after capturing 200 frames
        if count >= 200:
            break

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # After the loop release the cap object
    cap.release()

# Destroy all the windows
cv2.destroyAllWindows()