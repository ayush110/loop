import numpy as np
import cv2
# Load the trained model
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError  # Key addition
# --- CONFIGURATION ---
MODEL_PATH = 'donkeycar.h5'      # Path to your .h5 model
IMAGE_PATH = 'test2.jpg'      # Path to the test image
INPUT_SHAPE = (120, 160)     # (height, width) - adjust to your model's expected input
                             # DonkeyCar default: (120, 160)
# -----------------------

# Specify custom objects explicitly
model = load_model(
    'donkeycar.h5',
    custom_objects={'mse': MeanSquaredError()}  # Map 'mse' to actual class
)


# Read and preprocess the image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image at {IMAGE_PATH}")

# Resize and normalize image
img_resized = cv2.resize(img, INPUT_SHAPE[::-1])   # cv2 uses (width, height)
img_normalized = img_resized.astype(np.float32) / 255.0

# Add batch dimension
input_batch = np.expand_dims(img_normalized, axis=0)  # Shape: (1, H, W, 3)

# Predict steering
steering = model.predict(input_batch)
print(f"{steering}")
