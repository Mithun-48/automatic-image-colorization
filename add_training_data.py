from pathlib import Path
import csv
import requests
from PIL import Image
from io import BytesIO

# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATASET_DIR = Path(
    r"C:\Users\mithu\OneDrive\Documents\unsplash-research-dataset-lite-latest"
)

PROJECT_DIR = Path(
    r"C:\Users\mithu\OneDrive\Documents\5th sem\ML\automatic_image_colorization_project\automatic_image_colorization"
)

PHOTOS_FILE = DATASET_DIR / "photos.tsv000"
TRAIN_DIR = PROJECT_DIR / "data" / "train"

# Number of NEW images to add
TARGET_NEW_IMAGES = 350

# --------------------------------------------------
# CREATE TRAIN FOLDER
# --------------------------------------------------

TRAIN_DIR.mkdir(parents=True, exist_ok=True)

existing_images = list(TRAIN_DIR.glob("*.jpg"))

print(f"Existing training images: {len(existing_images)}")
print(f"Target new images: {TARGET_NEW_IMAGES}")

# Start numbering after the existing files
next_number = len(existing_images) + 1

# --------------------------------------------------
# NATURE / LANDSCAPE KEYWORDS
# --------------------------------------------------

keywords = [
    "landscape",
    "nature",
    "forest",
    "mountain",
    "mountains",
    "tree",
    "trees",
    "river",
    "lake",
    "waterfall",
    "ocean",
    "sea",
    "beach",
    "sky",
    "sunset",
    "sunrise",
    "valley",
    "desert",
    "park",
    "flower",
    "flowers",
    "plant",
    "plants",
    "green",
    "garden",
    "woods",
    "woodland",
    "cliff",
    "hill",
    "hills",
    "snow",
    "clouds"
]

# --------------------------------------------------
# READ UNSPLASH METADATA
# --------------------------------------------------

print("\nReading Unsplash metadata...")

downloaded = 0
checked = 0

with open(PHOTOS_FILE, "r", encoding="utf-8", errors="ignore") as f:

    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:

        if downloaded >= TARGET_NEW_IMAGES:
            break

        checked += 1

        # Combine useful text fields
        text = " ".join([
            row.get("photo_description", ""),
            row.get("photo_location_name", ""),
            row.get("ai_description", "")
        ]).lower()

        # Keep nature / landscape related images
        if not any(word in text for word in keywords):
            continue

        image_url = row.get("photo_image_url", "")

        if not image_url:
            continue

        output_name = f"image_{next_number:04d}.jpg"
        output_path = TRAIN_DIR / output_name

        try:

            print(
                f"Downloading {downloaded + 1}/{TARGET_NEW_IMAGES}: "
                f"{output_name}"
            )

            response = requests.get(
                image_url,
                timeout=30
            )

            if response.status_code != 200:
                print("  Download failed")
                continue

            image = Image.open(
                BytesIO(response.content)
            ).convert("RGB")

            # Resize while keeping aspect ratio
            max_width = 600

            if image.width > max_width:

                new_height = int(
                    image.height * max_width / image.width
                )

                image = image.resize(
                    (max_width, new_height),
                    Image.Resampling.LANCZOS
                )

            image.save(
                output_path,
                "JPEG",
                quality=95
            )

            downloaded += 1
            next_number += 1

        except Exception as e:

            print(f"  Error: {e}")
            continue

print("\n--------------------------------------")
print("DATASET EXPANSION COMPLETE")
print("--------------------------------------")
print(f"Metadata rows checked: {checked}")
print(f"New images downloaded: {downloaded}")
print(f"Total training images now: {len(list(TRAIN_DIR.glob('*.jpg')))}")
print("--------------------------------------")