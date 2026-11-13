
import csv
import hashlib
from pathlib import Path
from PIL import Image

SPLITS = {
    "train": "dataset/train.csv",
    "validation": "dataset/validation.csv",
    "test": "dataset/test.csv"
}

# Store which splits contain each unique image
image_records = {}

for split_name, csv_path in SPLITS.items():

    print(f"Checking {split_name} images...")

    with open(csv_path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_path = Path(row["image_path"])

            # Hash decoded image pixels, not JPEG file bytes
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                image_hash = hashlib.sha256(img.tobytes()).hexdigest()

            if image_hash not in image_records:
                image_records[image_hash] = {}

            image_records[image_hash].setdefault(
                split_name, []
            ).append(str(image_path))


# Find images appearing in more than one split
overlaps = []

for image_hash, splits in image_records.items():
    if len(splits) > 1:
        overlaps.append(splits)

print("\n========== DUPLICATE CHECK ==========")
print("Unique image contents:", len(image_records))
print("Images shared across splits:", len(overlaps))

for splits in overlaps[:10]:
    print("\nDuplicate found:")
    for split_name, paths in splits.items():
        print(f"{split_name}: {paths[0]}")

if not overlaps:
    print("\nNo exact duplicate images found across splits.")
else:
    print("\nPotential data leakage detected!")
