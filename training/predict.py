import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "traffic_sign_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

# ============================================================
# Load Model
# ============================================================

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    },
    compile=False,
    safe_mode=False
)

# ============================================================
# Load Class Names
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# ============================================================
# Prediction Function
# ============================================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image)

    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)

    predicted_class = np.argmax(predictions)

    confidence = float(np.max(predictions)) * 100

    return (
        class_names[predicted_class],
        confidence
    )

# ============================================================
# Test Prediction
# ============================================================

if __name__ == "__main__":

    test_image = input("Enter image path: ")

    prediction, confidence = predict_image(test_image)

    print("\nPrediction :", prediction)

    print(f"Confidence : {confidence:.2f}%")