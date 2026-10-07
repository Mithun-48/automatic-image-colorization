import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np
import requests


DATASET_DIR = Path(
    r"C:\Users\mithu\OneDrive\Documents\unsplash-research-dataset-lite-latest"
)

PROJECT_DIR = Path(
    r"C:\Users\mithu\OneDrive\Documents\5th sem\ML\automatic_image_colorization_project\automatic_image_colorization"
)

TSV_FILE = DATASET_DIR / "photos.tsv000"
TRAIN_DIR = PROJECT_DIR / "data" / "train"

TARGET_NEW_IMAGES = 1000

# The previous downloader checked 569 metadata rows.
# Skip those rows so we don't simply repeat the same batch.
SKIP_ROWS = 569

KEYWORDS = [
    "nature", "landscape", "forest", "mountain", "tree",
    "flower", "flowers", "garden", "lake", "river",
    "water", "ocean", "beach", "sky", "sunset",
    "sunrise", "grass", "plant", "plants", "leaf",
    "leaves", "rock", "rocks", "valley", "hill",
    "hills", "wood", "woods", "jungle", "park",
    "wildlife", "bird", "butterfly", "insect",
    "mushroom", "garden", "field", "meadow"
]


def is_nature_photo(row):
    text = " ".join([
        str(row.get("photo_description", "")),
        str(row.get("ai_description", "")),
        str(row.get("photo_url", ""))
    ]).lower()

    return any(
        re.search(r"\b" + re.escape(word) + r"\b", text)
        for word in KEYWORDS
    )


def download_image(url, output_path):
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if response.status_code != 200:
            return False

        data = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if image is None:
            return False

        height, width = image.shape[:2]

        if width > 600:
            new_height = int(height * 600 / width)
            image = cv2.resize(
                image,
                (600, new_height),
                interpolation=cv2.INTER_AREA
            )

        cv2.imwrite(str(output_path), image)
        return True

    except Exception:
        return False


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(TRAIN_DIR.glob("*.jpg"))
    next_number = len(existing) + 1

    downloaded = 0
    checked = 0

    print("Starting additional dataset download...")
    print(f"Existing training images: {len(existing)}")
    print(f"Target new images: {TARGET_NEW_IMAGES}")
    print(f"Skipping first {SKIP_ROWS} metadata rows...")
    print()

    with open(
        TSV_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row_number, row in enumerate(reader, start=1):

            if row_number <= SKIP_ROWS:
                continue

            checked += 1

            if downloaded >= TARGET_NEW_IMAGES:
                break

            if not is_nature_photo(row):
                continue

            url = row.get("photo_image_url", "")

            if not url:
                continue

            output_path = TRAIN_DIR / f"image_{next_number:04d}.jpg"

            if download_image(url, output_path):
                downloaded += 1
                next_number += 1

                print(
                    f"Downloaded {downloaded}/{TARGET_NEW_IMAGES}: "
                    f"{output_path.name}"
                )

            time.sleep(0.15)

    print()
    print("===================================")
    print("ADDITIONAL DATASET DOWNLOAD DONE")
    print("===================================")
    print(f"Metadata rows checked: {checked}")
    print(f"New images downloaded: {downloaded}")
    print(f"Total training images now: {len(list(TRAIN_DIR.glob('*.jpg')))}")


if __name__ == "__main__":
    main()