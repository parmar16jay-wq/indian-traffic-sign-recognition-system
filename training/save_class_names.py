import json
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset folder
DATASET_PATH = BASE_DIR / "Dataset" / "train"

# Models folder
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Get class names (folder names)
class_names = sorted(
    [folder.name for folder in DATASET_PATH.iterdir() if folder.is_dir()]
)

# Save class names
with open(MODELS_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("✅ Class names saved successfully!")
print(f"Total Classes: {len(class_names)}")
print(f"File saved at: {MODELS_DIR / 'class_names.json'}")