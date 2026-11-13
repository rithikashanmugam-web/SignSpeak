
import os
import numpy as np

# New ISL phrase dataset location
DATASET_PATH = r"C:\Users\sunda\Downloads\Dataset of Common phrases in Indian Sign Language\MP_Data"

EXPECTED_FRAMES = 30
EXPECTED_FEATURES = 1662

total_sequences = 0
total_frames = 0
class_counts = {}

invalid_files = []
invalid_shapes = []
incomplete_sequences = []

print("\n========== ISL PHRASE DATASET INSPECTION ==========\n")

for label in sorted(os.listdir(DATASET_PATH)):

    class_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(class_path):
        continue

    sequences = [
        folder for folder in os.listdir(class_path)
        if os.path.isdir(os.path.join(class_path, folder))
    ]

    class_counts[label] = len(sequences)
    class_frames = 0

    for sequence in sequences:

        sequence_path = os.path.join(class_path, sequence)

        files = [
            f for f in os.listdir(sequence_path)
            if f.endswith(".npy")
        ]

        class_frames += len(files)

        if len(files) != EXPECTED_FRAMES:
            incomplete_sequences.append(
                f"{label}/{sequence}: {len(files)} frames"
            )

        for file in files:
            file_path = os.path.join(sequence_path, file)

            try:
                data = np.load(file_path, allow_pickle=False)

                if data.shape != (EXPECTED_FEATURES,):
                    invalid_shapes.append(file_path)

                if not np.isfinite(data).all():
                    invalid_files.append(file_path)

            except Exception as error:
                invalid_files.append(f"{file_path}: {error}")

    total_sequences += len(sequences)
    total_frames += class_frames

    print(
        f"Class: {label:15} | "
        f"Sequences: {len(sequences):2} | "
        f"Frames: {class_frames:4}"
    )

print("\n========== DATASET SUMMARY ==========")

print("Total classes:", len(class_counts))
print("Total sequences:", total_sequences)
print("Total frames:", total_frames)

print("\n========== DATA QUALITY ==========")

print("Invalid files:", len(invalid_files))
print("Invalid shapes:", len(invalid_shapes))
print("Incomplete sequences:", len(incomplete_sequences))

print("\n========== INSPECTION COMPLETE ==========")
