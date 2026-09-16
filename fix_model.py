import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OLD_MODEL = BASE_DIR / "models" / "traffic_sign_model.keras"
NEW_MODEL = BASE_DIR / "models" / "traffic_sign_model_fixed.keras"

print("Loading model...")

model = tf.keras.models.load_model(
    OLD_MODEL,
    custom_objects={
        "preprocess_input": preprocess_input
    },
    compile=False,
    safe_mode=False
)

print("Saving fixed model...")

model.save(NEW_MODEL)

print("\n✅ Done!")
print(f"Saved to: {NEW_MODEL}")