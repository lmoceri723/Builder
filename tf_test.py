import cv2
from builder import LegoClassifier

def test_color_recognition():
    # Create an instance of LegoClassifier
    lego_classifier = LegoClassifier("data/final_train_imgs")

    # Use opencv to capture video
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            break

        # Get the color of the frame
        color = lego_classifier.find_color(frame)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

# Run the test
test_color_recognition()