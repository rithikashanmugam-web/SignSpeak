
import os
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 1. Configuration

DEVICE = torch.device("cpu")

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

CHECKPOINT_PATH = "models/last_checkpoint.pth"
BEST_MODEL_PATH = "models/sign_model.pth"

os.makedirs("models", exist_ok=True)

print("Using device:", DEVICE)

# 2. Image preprocessing

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# 3. Custom Dataset

class SignDataset(Dataset):

    def __init__(self, csv_file, class_to_index):

        self.data = []

        with open(csv_file, "r", newline="") as file:

            reader = csv.DictReader(file)

            for row in reader:
                self.data.append((
                    row["image_path"],
                    class_to_index[row["label"]]
                ))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image_path, label = self.data[index]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label

# 4. Read class names

with open("dataset/train.csv", "r", newline="") as file:

    reader = csv.DictReader(file)

    class_names = sorted({
        row["label"] for row in reader
    })

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}

print("Number of classes:", len(class_names))

# 5. Load datasets

train_dataset = SignDataset(
    "dataset/train.csv", class_to_index
)

val_dataset = SignDataset(
    "dataset/validation.csv", class_to_index
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# 6. Build CNN

class SignCNN(nn.Module):

    def __init__(self, num_classes):

        super(SignCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

# 7. Initialize model

model = SignCNN(num_classes=len(class_names))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# 8. Resume from checkpoint if available

start_epoch = 0
best_accuracy = 0.0

if os.path.exists(CHECKPOINT_PATH):

    print("\nCheckpoint found!")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if checkpoint["class_names"] != class_names:
        raise ValueError("Checkpoint classes do not match dataset!")

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]
    best_accuracy = checkpoint["best_accuracy"]

    print("Resuming from epoch:", start_epoch + 1)

else:

    print("\nNo checkpoint found. Starting fresh!")

# 9. Training loop

for epoch in range(start_epoch, EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    # Validation

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    # Display results

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    print(
        f"Training Loss: "
        f"{running_loss / len(train_loader):.4f}"
    )

    print(
        f"Training Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss / len(val_loader):.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    # Save best model

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save({

            "model_state_dict": model.state_dict(),
            "class_names": class_names

        }, BEST_MODEL_PATH)

        with open("models/class_names.json", "w") as file:
            json.dump(class_names, file)

        print("Best model saved!")

    # Save checkpoint after every completed epoch

    torch.save({

        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": best_accuracy,
        "class_names": class_names

    }, CHECKPOINT_PATH)

    print("Training checkpoint saved!")

print("\nTraining completed!")
print("Best validation accuracy:", round(best_accuracy, 2), "%")