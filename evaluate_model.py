
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv


# 1. Configuration
DEVICE = torch.device("cpu")
BATCH_SIZE = 32

# 2. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# 3. Dataset
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image_path, label = self.data[index]

        image = Image.open(image_path).convert("RGB")
        image = transform(image)

        return image, label


# 4. Load saved model
checkpoint = torch.load(
    "models/sign_model.pth",
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}


# 5. Recreate CNN architecture
class SignCNN(nn.Module):

    def __init__(self, num_classes):

        super(SignCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.features(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)


model = SignCNN(len(class_names))

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()


# 6. Load test dataset
test_dataset = SignDataset(
    "dataset/test.csv",
    class_to_index
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# 7. Evaluate model
correct = 0
total = 0

class_correct = [0] * len(class_names)
class_total = [0] * len(class_names)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):

            label = labels[i].item()

            class_total[label] += 1

            if predicted[i] == labels[i]:
                class_correct[label] += 1


# 8. Display results
accuracy = 100 * correct / total

print("\n========== TEST RESULTS ==========")

print("Total test images:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", total - correct)

print(f"Test Accuracy: {accuracy:.2f}%")

print("\n========== CLASS-WISE ACCURACY ==========")

for i, name in enumerate(class_names):

    if class_total[i] > 0:

        class_accuracy = (
            100 * class_correct[i] / class_total[i]
        )

        print(f"{name}: {class_accuracy:.2f}%")
