import os
import cv2

# Start video capture from the default camera (0 represents the first camera device)
cap = cv2.VideoCapture(0)

# Directory where images for each letter are saved
directory = 'Image/'

# Continuously capture frames from the webcam
while True:
    _, frame = cap.read()  # Read a frame from the video capture object

    # Dictionary to count the number of images in each letter subdirectory
    count = {
        'a': len(os.listdir(directory + "/A")),
        'b': len(os.listdir(directory + "/B")),
        'c': len(os.listdir(directory + "/C")),
    }

    # Get dimensions of the captured frame
    row = frame.shape[1]
    col = frame.shape[0]

    # Draw a white rectangle on the frame to mark the Region of Interest (ROI)
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

    # Display the main capture window and the ROI window
    cv2.imshow("data", frame)
    cv2.imshow("ROI", frame[40:400, 0:300])

    # Crop the frame to show only the ROI area
    frame = frame[40:400, 0:300]

    # Check for key press to save the frame to corresponding letter folders
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory + 'A/' + str(count['a']) + '.png', frame)
    elif interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory + 'B/' + str(count['b']) + '.png', frame)
    elif interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory + 'C/' + str(count['c']) + '.png', frame)

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
