
import numpy as np
import json

# Load confusion matrix
confusion = np.load("models/confusion_matrix.npy")

# Load class labels
with open("phrase_dataset/class_names.json") as file:
    class_names = json.load(file)

print("\n========== MISCLASSIFICATION ANALYSIS ==========\n")

mistakes = []

for actual_index in range(len(class_names)):

    for predicted_index in range(len(class_names)):

        if actual_index == predicted_index:
            continue

        count = confusion[actual_index, predicted_index]

        if count > 0:
            mistakes.append((
                count,
                class_names[actual_index],
                class_names[predicted_index]
            ))

# Show most frequent confusion pairs
mistakes.sort(reverse=True)

print("Actual sign -> Predicted sign\n")

for count, actual, predicted in mistakes:
    print(f"{actual:15} -> {predicted:15} | {count} time(s)")

print("\n========== MOST DIFFICULT CLASSES ==========\n")

for i, label in enumerate(class_names):

    total = confusion[i].sum()
    correct = confusion[i, i]
    incorrect = total - correct

    print(
        f"{label:15} | "
        f"Correct: {correct}/{total} | "
        f"Incorrect: {incorrect}"
    )

print("\nAnalysis completed!")
