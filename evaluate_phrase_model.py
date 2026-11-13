
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Configuration
DATA_PATH = "phrase_dataset"
MODEL_PATH = "models/isl_phrase_lstm.pth"

BATCH_SIZE = 16
DEVICE = torch.device("cpu")


# Same architecture used during training
class ISLPhraseLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)

        return self.fc(last_output)


# Load saved checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]

model = ISLPhraseLSTM(
    checkpoint["input_size"],
    checkpoint["hidden_size"],
    checkpoint["num_classes"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Model loaded successfully!")
print("Number of classes:", len(class_names))


# Load test dataset
X_test = np.load(
    os.path.join(DATA_PATH, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_PATH, "y_test.npy")
)

X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.int64))

test_dataset = TensorDataset(X_test, y_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Evaluate model
all_predictions = []
all_actual = []

correct = 0
total = 0

with torch.no_grad():

    for inputs, labels in test_loader:

        outputs = model(inputs)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_predictions.extend(predictions.numpy())
        all_actual.extend(labels.numpy())


# Overall accuracy
accuracy = (correct / total) * 100

print("\n========== TEST RESULTS ==========")

print("Correct predictions:", correct)
print("Total samples:", total)
print(f"Test Accuracy: {accuracy:.2f}%")


# Per-class accuracy
all_predictions = np.array(all_predictions)
all_actual = np.array(all_actual)

print("\n========== CLASS-WISE ACCURACY ==========")

for class_index, class_name in enumerate(class_names):

    mask = all_actual == class_index

    class_total = mask.sum()

    class_correct = (
        all_predictions[mask] == all_actual[mask]
    ).sum()

    class_accuracy = (
        class_correct / class_total
    ) * 100 if class_total > 0 else 0

    print(
        f"{class_name:15} | "
        f"{class_correct}/{class_total} | "
        f"{class_accuracy:.2f}%"
    )


# Confusion matrix
num_classes = len(class_names)

confusion = np.zeros(
    (num_classes, num_classes),
    dtype=np.int64
)

for actual, predicted in zip(all_actual, all_predictions):
    confusion[actual, predicted] += 1

np.save(
    "models/confusion_matrix.npy",
    confusion
)

print("\nConfusion matrix saved!")
print("Evaluation completed!")
