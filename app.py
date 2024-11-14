from function import *  # Import custom functions for mediapipe and other operations
from keras.utils import to_categorical  # For categorical data processing
from keras.models import model_from_json  # To load model architecture from JSON
from keras.layers import LSTM, Dense  # Layers for neural network
from keras.callbacks import TensorBoard  # For monitoring and visualization of training

# Load the trained model architecture from JSON file
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")  # Load weights for the model

# Set colors for different actions in visualization
colors = []
for i in range(0, 20):
    colors.append((245, 117, 16))  # Add a color for each action

# Function for visualizing the probability of each action
def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()  # Copy the input frame for visualization
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)  # Draw a rectangle for each probability
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Label each rectangle
    return output_frame  # Return frame with visualization

# Detection and display variables
sequence = []  # Store detected keypoints sequences
sentence = []  # Detected actions in a sentence form
accuracy = []  # Prediction accuracies
predictions = []  # Store action predictions
threshold = 0.8  # Confidence threshold for action detection

cap = cv2.VideoCapture(0)  # Capture video from the default camera

# Initialize mediapipe for hand tracking
with mp_hands.Hands(
    model_complexity=0,  # Simpler model for faster performance
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # Main loop to capture each frame
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video capture

        # Process and crop the frame for hand detection
        cropframe = frame[40:400, 0:300]  # Define the region of interest
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)  # Draw a rectangle around the region
        image, results = mediapipe_detection(cropframe, hands)  # Run hand detection

        # Extract keypoints from the detected hand landmarks
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)  # Append the keypoints to the sequence
        sequence = sequence[-30:]  # Keep the last 30 frames only

        try:
            # If we have a full sequence of frames, make a prediction
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]  # Predict on the sequence
                predictions.append(np.argmax(res))  # Add prediction to list

                # Check if the prediction is consistent and above the threshold
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            # Add the action to sentence if it's different from the last
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)] * 100))
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)] * 100))

                if len(sentence) > 1:
                    # Keep only the most recent action
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

        except Exception as e:
            pass  # Handle exceptions during prediction
        
        # Display the prediction on the frame
        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: -" + ' '.join(sentence) + ''.join(accuracy), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', frame)  # Show the frame

        if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()  # Release the video capture
    cv2.destroyAllWindows()  # Close the display window