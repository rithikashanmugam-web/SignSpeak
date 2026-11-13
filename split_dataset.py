
import os
import csv
import random

DATASET_PATH = "dataset/data"
OUTPUT_PATH = "dataset"

random.seed(42)

train_data = []
val_data = []
test_data = []

classes = sorted(os.listdir(DATASET_PATH))

for label in classes:

    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for image in train_images:
        train_data.append([label, os.path.join(folder, image)])

    for image in val_images:
        val_data.append([label, os.path.join(folder, image)])

    for image in test_images:
        test_data.append([label, os.path.join(folder, image)])

def save_csv(filename, data):

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "image_path"])
        writer.writerows(data)

save_csv("dataset/train.csv", train_data)
save_csv("dataset/validation.csv", val_data)
save_csv("dataset/test.csv", test_data)

print("Dataset splitting completed!")

print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Testing images:", len(test_data))

