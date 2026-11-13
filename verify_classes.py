
import os

DATASET_PATH = r"C:\Users\sunda\Downloads\Dataset of Common phrases in Indian Sign Language\MP_Data"

print("\n========== FINAL CLASS VERIFICATION ==========\n")

classes = []

for label in sorted(os.listdir(DATASET_PATH)):
    path = os.path.join(DATASET_PATH, label)

    if os.path.isdir(path):
        sequences = [
            folder for folder in os.listdir(path)
            if os.path.isdir(os.path.join(path, folder))
        ]

        valid_sequences = 0

        for sequence in sequences:
            sequence_path = os.path.join(path, sequence)

            frames = [
                f for f in os.listdir(sequence_path)
                if f.endswith(".npy")
            ]

            if len(frames) == 30:
                valid_sequences += 1

        if valid_sequences > 0:
            classes.append(label)

        print(
            f"{label:15} | "
            f"Valid sequences: {valid_sequences}"
        )

print("\nTotal usable classes:", len(classes))
print("Total usable sequences:", len(classes) * 30)

print("\nClass labels:")
print(classes)
