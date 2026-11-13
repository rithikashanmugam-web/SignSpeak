
import os
import random
from PIL import Image

DATASET_PATH = "dataset/data"

random.seed(42)

total_checked = 0
corrupted = 0
dimensions = {}
formats = {}

for label in sorted(os.listdir(DATASET_PATH)):

    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Select maximum 20 images per class
    selected = random.sample(images, min(20, len(images)))

    for filename in selected:

        path = os.path.join(folder, filename)

        try:
            with Image.open(path) as img:

                img.verify()

                total_checked += 1

                dimensions[img.size] = dimensions.get(img.size, 0) + 1

                formats[img.format] = formats.get(img.format, 0) + 1

        except Exception:
            corrupted += 1

print("\n===== IMAGE QUALITY REPORT =====")
print("Images checked:", total_checked)
print("Corrupted images:", corrupted)

print("\nImage dimensions:")
for size, count in dimensions.items():
    print(size, ":", count)

print("\nImage formats:")
for fmt, count in formats.items():
    print(fmt, ":", count)
