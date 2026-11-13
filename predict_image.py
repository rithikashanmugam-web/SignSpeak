
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import json
import sys

# 1. Configuration
DEVICE = torch.device("cpu")
IMAGE_SIZE = 128

# 2. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# 3. Recreate the same CNN architecture used during training
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
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# 4. Load trained model
checkpoint = torch.load(
    "models/sign_model.pth",
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]

model = SignCNN(len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# 5. Predict an image
def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

    predicted_label = class_names[predicted_index.item()]
    confidence_score = confidence.item() * 100

    print("\n========== SIGN PREDICTION ==========")
    print("Predicted Sign:", predicted_label)
    print(f"Confidence: {confidence_score:.2f}%")
    print("=====================================")


# 6. Get image path from command line
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path>")
    else:
        predict_image(sys.argv[1])
