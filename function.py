import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize mediapipe utilities for drawing and hand detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Perform mediapipe detection on an image using a specified model
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color for mediapipe
    image.flags.writeable = False  # Improve performance by disabling write access
    results = model.process(image)  # Process the image to detect hands
    image.flags.writeable = True  # Re-enable write access to the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    return image, results  # Return the processed image and detection results

# Draw landmarks and hand connections on the image
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

# Extract keypoints from the detected landmarks or return zeros if no landmarks are detected
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
        return rh
    return np.zeros(21 * 3)  # Return zero array if no hand landmarks are found

# Define paths and parameters for data collection
DATA_PATH = os.path.join('MP_Data')  # Directory for storing data
actions = np.array(['A', 'B', 'C'])  # Actions corresponding to hand gestures
no_sequences = 30  # Number of sequences for each action
sequence_length = 30  # Length of each sequence for action
