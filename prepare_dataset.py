import csv
import os
import time
from io import BytesIO

import requests
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

UNSPLASH_TSV = r"C:\Users\mithu\OneDrive\Documents\unsplash-research-dataset-lite-latest\photos.tsv000"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(PROJECT_DIR, "data", "train")
TEST_DIR = os.path.join(PROJECT_DIR, "data", "test")
TEST_ORIGINAL_DIR = os.path.join(PROJECT_DIR, "data", "test_original")

TRAIN_COUNT = 150
TEST_COUNT = 30
IMAGE_WIDTH = 600

NATURE_KEYWORDS = [
    "nature", "landscape", "mountain", "mountains", "forest", "tree", "trees",
    "river", "lake", "waterfall", "ocean", "sea", "beach", "coast", "coastal",
    "valley", "hill", "hills", "sky", "sunset", "sunrise", "desert", "canyon",
    "rock", "rocks", "water", "snow", "snowy", "flower", "flowers", "wildlife",
    "outdoor", "outdoors", "park", "national park", "meadow", "field", "rural",
    "countryside", "jungle", "woodland", "cliff", "island"
]


def contains_nature_keyword(row):
    fields = [
        row.get("photo_description", ""),
        row.get("ai_description", ""),
        row.get("photo_location_name", ""),
        row.get("photo_location_city", ""),
        row.get("photo_location_country", ""),
        row.get("ai_primary_landmark_name", ""),
    ]
    text = " ".join(fields).lower()
    return any(keyword in text for keyword in NATURE_KEYWORDS)


def clean_filename(index):
    return f"image_{index:04d}.jpg"


def download_image(url):
    separator = "&" if "?" in url else "?"
    url = url + separator + "auto=format&fit=max&w=800&q=80"

    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Automatic-Image-Colorization-Project/1.0"},
    )
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def prepare_image(image):
    width, height = image.size
    if width > IMAGE_WIDTH:
        new_height = int(height * IMAGE_WIDTH / width)
        image = image.resize((IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
    return image


def save_grayscale(image, path):
    image.convert("L").save(path, quality=95)


def main():
    print("=" * 60)
    print("AUTOMATIC IMAGE COLORIZATION")
    print("Unsplash Lite Dataset Preparation")
    print("=" * 60)

    if not os.path.exists(UNSPLASH_TSV):
        print("\nERROR: Could not find:")
        print(UNSPLASH_TSV)
        return

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(TEST_ORIGINAL_DIR, exist_ok=True)

    print("\nDataset metadata found.")
    print("Selecting nature / landscape photographs...")

    selected = []

    with open(UNSPLASH_TSV, "r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            url = row.get("photo_image_url", "").strip()
            if not url or not contains_nature_keyword(row):
                continue

            try:
                width = int(row.get("photo_width", "0"))
                height = int(row.get("photo_height", "0"))
            except ValueError:
                continue

            if width < 500 or height < 300:
                continue

            selected.append(row)

            if len(selected) >= TRAIN_COUNT + TEST_COUNT:
                break

    print(f"\nFound {len(selected)} suitable photographs.")

    if len(selected) < TRAIN_COUNT + TEST_COUNT:
        print(f"ERROR: Need {TRAIN_COUNT + TEST_COUNT} images but only found {len(selected)}.")
        return

    print("\nDownloading images...")
    print(f"Training images : {TRAIN_COUNT}")
    print(f"Testing images  : {TEST_COUNT}")
    print()

    train_done = 0
    test_done = 0
    failed = 0

    for index, row in enumerate(selected):
        try:
            print(f"[{index + 1}/{len(selected)}] Downloading...")

            image = prepare_image(download_image(row["photo_image_url"].strip()))

            if train_done < TRAIN_COUNT:
                filename = clean_filename(train_done + 1)
                image.save(os.path.join(TRAIN_DIR, filename), "JPEG", quality=90)
                train_done += 1
            else:
                filename = clean_filename(test_done + 1)
                image.save(
                    os.path.join(TEST_ORIGINAL_DIR, filename),
                    "JPEG",
                    quality=90,
                )
                save_grayscale(image, os.path.join(TEST_DIR, filename))
                test_done += 1

            time.sleep(0.1)

        except Exception as error:
            failed += 1
            print(f"  FAILED: {error}")

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nTraining images : {train_done}")
    print(f"Testing images  : {test_done}")
    print(f"Failed downloads: {failed}")

    if train_done == TRAIN_COUNT and test_done == TEST_COUNT:
        print("\nSUCCESS! The dataset is ready for model training.")
    else:
        print("\nWARNING: Not all images were downloaded.")


if __name__ == "__main__":
    main()
