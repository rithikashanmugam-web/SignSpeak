
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = "phrase_dataset"
MODEL_PATH = "models"

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.001
PATIENCE = 7

INPUT_SIZE = 1662
HIDDEN_SIZE = 64

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")

os.makedirs(MODEL_PATH, exist_ok=True)

print("Using device:", DEVICE)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading training data...")

X_train = np.load(
    os.path.join(DATA_PATH, "X_train.npy")
)

y_train = np.load(
    os.path.join(DATA_PATH, "y_train.npy")
)

X_val = np.load(
    os.path.join(DATA_PATH, "X_val.npy")
)

y_val = np.load(
    os.path.join(DATA_PATH, "y_val.npy")
)

with open(
    os.path.join(DATA_PATH, "class_names.json")
) as file:
    class_names = json.load(file)

NUM_CLASSES = len(class_names)

print("Classes:", NUM_CLASSES)
print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)


# ==========================================
# 3. CONVERT TO PYTORCH TENSORS
# ==========================================

X_train_tensor = torch.from_numpy(
    X_train.astype(np.float32, copy=False)
)

y_train_tensor = torch.from_numpy(
    y_train.astype(np.int64, copy=False)
)

X_val_tensor = torch.from_numpy(
    X_val.astype(np.float32, copy=False)
)

y_val_tensor = torch.from_numpy(
    y_val.astype(np.int64, copy=False)
)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
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


# ==========================================
# 4. BUILD LSTM MODEL
# ==========================================

class ISLPhraseLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        # Use the final time-step output
        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)

        predictions = self.fc(last_output)

        return predictions


model = ISLPhraseLSTM(
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_CLASSES
).to(DEVICE)

print("\nLSTM model created successfully!")


# ==========================================
# 5. LOSS AND OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 6. TRAINING AND VALIDATION
# ==========================================

best_val_accuracy = 0.0
best_val_loss = float("inf")
epochs_without_improvement = 0

checkpoint_path = os.path.join(
    MODEL_PATH,
    "isl_phrase_lstm.pth"
)

for epoch in range(EPOCHS):

    # ----------------------
    # TRAINING
    # ----------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_accuracy = (
        train_correct / train_total
    ) * 100

    train_loss /= len(train_loader)


    # ----------------------
    # VALIDATION
    # ----------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_accuracy = (
        val_correct / val_total
    ) * 100

    val_loss /= len(val_loader)


    # ----------------------
    # DISPLAY RESULTS
    # ----------------------

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )


    # ----------------------
    # SAVE BEST MODEL
    # ----------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_classes": NUM_CLASSES
            },
            checkpoint_path
        )

        epochs_without_improvement = 0

        print("Best model saved!")

    elif val_accuracy == best_val_accuracy:

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "input_size": INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_classes": NUM_CLASSES
                },
                checkpoint_path
            )

            epochs_without_improvement = 0
            print("Best model updated!")

        else:
            epochs_without_improvement += 1

    else:
        epochs_without_improvement += 1


    # ----------------------
    # EARLY STOPPING
    # ----------------------

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered!")

        break


print("\n========== TRAINING COMPLETE ==========")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print("Saved model:", checkpoint_path)
