from function import *  # Import custom helper functions
from sklearn.model_selection import train_test_split  # To split data for training and testing
from keras.utils import to_categorical  # For encoding labels
from keras.models import Sequential  # Sequential model architecture
from keras.layers import LSTM, Dense  # Layers for the model
from keras.callbacks import TensorBoard  # For logging and monitoring training progress
import numpy as np
import os

# Map actions to numeric labels
label_map = {label: num for num, label in enumerate(actions)}

# Load and pad sequences for model input
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if os.path.exists(npy_path):
                res = np.load(npy_path, allow_pickle=True)  # Load frame keypoints
                window.append(res)
            else:
                window.append(np.zeros(21 * 3))  # Pad missing frames with zero keypoints

        sequences.append(window)
        labels.append(label_map[action])

# Prepare training and testing data
X = np.array(sequences)  # Input data
y = to_categorical(labels).astype(int)  # Labels in categorical format
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y)

log_dir = os.path.join('Logs')  # Log directory for TensorBoard
tb_callback = TensorBoard(log_dir=log_dir)  # Initialize TensorBoard callback

# Define the LSTM model architecture
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 21 * 3)),  # First LSTM layer
    LSTM(128, return_sequences=True, activation='relu'),  # Second LSTM layer
    LSTM(64, return_sequences=False, activation='relu'),  # Third LSTM layer
    Dense(64, activation='relu'),  # Fully connected layer with 64 units
    Dense(32, activation='relu'),  # Fully connected layer with 32 units
    Dense(len(actions), activation='softmax')  # Output layer with softmax for action classification
])

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model with training data and TensorBoard callback
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))
model.summary()  # Print the model architecture summary

# Save the model architecture and weights
model_json = model.to_json()  # Serialize model to JSON
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')  # Save model weights
print("Model saved successfully.")
