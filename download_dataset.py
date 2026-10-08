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


print("Downloading landscape dataset...")

dataset = load_dataset(
    "ferrariedhgs/Nature-Landscape",
    split="train"
)

dataset = dataset.shuffle(seed=42)


train_images = 850
test_images = 150


for i in range(train_images):

    image = dataset[i]["image"].convert("RGB")

    image = np.array(image)

    path = os.path.join(
        train_folder,
        f"image_{i + 1:04d}.jpg"
    )

    cv2.imwrite(
        path,
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )


for i in range(test_images):

    image = dataset[train_images + i]["image"].convert("RGB")

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

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    cv2.imwrite(
        gray_path,
        gray
    )


print()
print("Dataset ready.")
print("Training images:", train_images)
print("Test images:", test_images)