
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = "dataset/data"

random.seed(42)

classes = sorted(os.listdir(DATASET_PATH))

fig, axes = plt.subplots(5, 7, figsize=(15, 12))

axes = axes.flatten()

for i, label in enumerate(classes):

    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    selected_image = random.choice(images)

    image_path = os.path.join(folder_path, selected_image)

    img = Image.open(image_path).convert("RGB")

    axes[i].imshow(img)
    axes[i].set_title(f"Class: {label}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()
