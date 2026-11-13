
import os
import json
import numpy as np

# Dataset location
DATASET_PATH = r"C:\Users\sunda\Downloads\Dataset of Common phrases in Indian Sign Language\MP_Data"

# Output location
OUTPUT_PATH = "phrase_dataset"

# Dataset configuration
FRAMES = 30
FEATURES = 1662

TRAIN_PER_CLASS = 24
VAL_PER_CLASS = 3
TEST_PER_CLASS = 3

RANDOM_SEED = 42

os.makedirs(OUTPUT_PATH, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)

# Find usable classes
class_names = []

for label in sorted(os.listdir(DATASET_PATH)):
    class_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(class_path):
        continue

    valid_sequences = []

    for sequence in sorted(
        os.listdir(class_path),
        key=lambda x: int(x) if x.isdigit() else x
    ):
        sequence_path = os.path.join(class_path, sequence)

        if not os.path.isdir(sequence_path):
            continue

        frame_files = [
            f for f in os.listdir(sequence_path)
            if f.endswith(".npy")
        ]

        if len(frame_files) == FRAMES:
            valid_sequences.append(sequence)

    if len(valid_sequences) >= 30:
        class_names.append(label)

print("Usable classes:", len(class_names))

# Allocate memory for the datasets
total_train = len(class_names) * TRAIN_PER_CLASS
total_val = len(class_names) * VAL_PER_CLASS
total_test = len(class_names) * TEST_PER_CLASS

X_train = np.empty((total_train, FRAMES, FEATURES), dtype=np.float32)
X_val = np.empty((total_val, FRAMES, FEATURES), dtype=np.float32)
X_test = np.empty((total_test, FRAMES, FEATURES), dtype=np.float32)

y_train = np.empty(total_train, dtype=np.int64)
y_val = np.empty(total_val, dtype=np.int64)
y_test = np.empty(total_test, dtype=np.int64)

train_index = 0
val_index = 0
test_index = 0

# Process each class
for label_index, label in enumerate(class_names):

    class_path = os.path.join(DATASET_PATH, label)

    sequences = [
        folder for folder in os.listdir(class_path)
        if os.path.isdir(os.path.join(class_path, folder))
        and folder.isdigit()
    ]

    sequences = sorted(sequences, key=int)

    valid_sequences = []

    for sequence in sequences:
        sequence_path = os.path.join(class_path, sequence)

        frame_files = [
            f for f in os.listdir(sequence_path)
            if f.endswith(".npy")
        ]

        if len(frame_files) == FRAMES:
            valid_sequences.append(sequence)

    if len(valid_sequences) != 30:
        raise ValueError(
            f"{label} has {len(valid_sequences)} valid sequences, expected 30"
        )

    # Shuffle sequence order reproducibly
    rng.shuffle(valid_sequences)

    train_sequences = valid_sequences[:24]
    val_sequences = valid_sequences[24:27]
    test_sequences = valid_sequences[27:30]

    def load_sequence(sequence):
        sequence_path = os.path.join(class_path, sequence)

        frames = []

        for frame_number in range(FRAMES):
            file_path = os.path.join(
                sequence_path,
                f"{frame_number}.npy"
            )

            data = np.load(file_path, allow_pickle=False)

            if data.shape != (FEATURES,):
                raise ValueError(f"Invalid shape: {file_path}")

            if not np.isfinite(data).all():
                raise ValueError(f"Invalid numerical values: {file_path}")

            frames.append(data.astype(np.float32))

        return np.stack(frames)

    # Training sequences
    for sequence in train_sequences:
        X_train[train_index] = load_sequence(sequence)
        y_train[train_index] = label_index
        train_index += 1

    # Validation sequences
    for sequence in val_sequences:
        X_val[val_index] = load_sequence(sequence)
        y_val[val_index] = label_index
        val_index += 1

    # Testing sequences
    for sequence in test_sequences:
        X_test[test_index] = load_sequence(sequence)
        y_test[test_index] = label_index
        test_index += 1

    print(f"Processed: {label}")

# Save datasets
np.save(os.path.join(OUTPUT_PATH, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_PATH, "y_train.npy"), y_train)

np.save(os.path.join(OUTPUT_PATH, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_PATH, "y_val.npy"), y_val)

np.save(os.path.join(OUTPUT_PATH, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_PATH, "y_test.npy"), y_test)

# Save class names
with open(
    os.path.join(OUTPUT_PATH, "class_names.json"),
    "w"
) as file:
    json.dump(class_names, file, indent=4)

print("\n========== DATASET PREPARATION COMPLETE ==========")

print("Classes:", len(class_names))
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nSaved successfully in:", OUTPUT_PATH)
