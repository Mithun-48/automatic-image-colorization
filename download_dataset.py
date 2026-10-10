import os
import cv2
import numpy as np
from datasets import load_dataset


train_folder = "data/train"
test_folder = "data/test"
original_folder = "data/test_original"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(original_folder, exist_ok=True)

print("Loading Intel Natural Scene dataset...")

dataset = load_dataset("sfarrukhm/intel-image-classification")

natural_labels = {1, 2, 3, 4}

train_data = dataset["train"].filter(
    lambda x: x["label"] in natural_labels
)

test_data = dataset["test"].filter(
    lambda x: x["label"] in natural_labels
)

train_data = train_data.shuffle(seed=42)
test_data = test_data.shuffle(seed=42)

train_count = min(9000, len(train_data))
test_count = min(150, len(test_data))

print("Natural training images:", train_count)
print("Natural test images:", test_count)

for i in range(train_count):
    image = train_data[i]["image"].convert("RGB")
    image = np.array(image)

    path = os.path.join(
        train_folder,
        f"image_{i + 1:04d}.jpg"
    )

    cv2.imwrite(
        path,
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )

for i in range(test_count):
    image = test_data[i]["image"].convert("RGB")
    image = np.array(image)

    original_path = os.path.join(
        original_folder,
        f"image_{i + 1:04d}.jpg"
    )

    gray_path = os.path.join(
        test_folder,
        f"image_{i + 1:04d}.jpg"
    )

    cv2.imwrite(
        original_path,
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    cv2.imwrite(
        gray_path,
        gray
    )

print()
print("Dataset ready.")
print("Training images:", train_count)
print("Test images:", test_count)