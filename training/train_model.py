import os
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "Dataset" / "train"

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Project Folder :", BASE_DIR)
print("Dataset Folder :", DATASET_PATH)
print("Dataset Exists :", DATASET_PATH.exists())
print("=" * 60)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset folder not found!\nExpected location:\n{DATASET_PATH}"
    )

# ============================================================
# SETTINGS
# ============================================================

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20

# ============================================================
# LOAD DATASET
# ============================================================

train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

validation_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nDataset Loaded Successfully")
print("-" * 60)
print("Total Classes :", len(class_names))
print()

for i, name in enumerate(class_names):
    print(f"{i+1}. {name}")

# ============================================================
# OPTIMIZE DATASET
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# ============================================================
# LOAD MOBILENETV2
# ============================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

model = tf.keras.Sequential([
    data_augmentation,

    tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1),

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax",
    ),
])

model.build((None, 224, 224, 3))

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n")
print("=" * 60)
print("MODEL CREATED SUCCESSFULLY")
print("=" * 60)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

checkpoint = ModelCheckpoint(
    filepath=str(MODELS_DIR / "traffic_sign_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1,
)

# ============================================================
# TRAIN MODEL
# ============================================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr,
    ],
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(MODELS_DIR / "final_traffic_sign_model.keras")

print("\nFinal Model Saved Successfully")

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(MODELS_DIR / "accuracy.png")
plt.close()

# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(MODELS_DIR / "loss.png")
plt.close()

print("\nGraphs Saved Successfully")
print("Accuracy Graph :", MODELS_DIR / "accuracy.png")
print("Loss Graph     :", MODELS_DIR / "loss.png")