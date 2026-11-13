# 🤟 SignSpeak – Indian Sign Language Recognition Using Deep Learning

### Bridging Communication Through Artificial Intelligence

SignSpeak is an AI-powered Indian Sign Language (ISL) recognition system that uses Deep Learning and Computer Vision to recognize static hand signs and dynamic sign language phrases.

The project combines **Convolutional Neural Networks (CNN)** for static sign classification and **Long Short-Term Memory (LSTM)** networks for dynamic phrase recognition, enabling two different approaches to ISL recognition.

---

## 📌 Project Overview

Communication is essential for everyone. However, communication barriers can make everyday interactions challenging for people who use sign language.

SignSpeak explores how Artificial Intelligence and Computer Vision can help recognize Indian Sign Language gestures and phrases.

The system consists of two independent recognition modules:

- **Static Sign Recognition:** Identifies ISL alphabets (A–Z) and digits (1–9) from images using a CNN model.
- **Dynamic Phrase Recognition:** Recognizes common ISL phrases from sequences of body, face, and hand landmarks using an LSTM model.

## ✨ Key Features

- 🤟 Recognition of 35 static sign classes (A–Z and digits 1–9).
- 🗣️ Recognition of 40 common ISL phrase classes.
- 🧠 CNN-based static sign classification.
- 🔄 LSTM-based sequential phrase recognition.
- 📷 Real-time webcam-based phrase recognition.
- ✋ Hand, face, and pose landmark detection using MediaPipe.
- 📊 Model evaluation and confusion matrix analysis.
- ⚡ CPU-compatible implementation.

## 🛠️ Technologies Used

| Technology | Application |
|---|---|
| Python | Core programming language |
| PyTorch | Deep Learning model development |
| Torchvision | Image preprocessing |
| OpenCV | Image processing and webcam access |
| MediaPipe | Landmark detection |
| NumPy | Numerical computation |
| Pillow | Image processing |
| Matplotlib | Data visualization |

## 🏗️ System Architecture

SignSpeak consists of two independent recognition pipelines.

### 1. Static Sign Recognition – CNN

```text
Input Sign Image
       |
Image Preprocessing
       |
Resize to 128 × 128
       |
Normalization
       |
   CNN Model
       |
35-Class Classification
       |
Predicted Alphabet / Digit
```

### 2. Dynamic Phrase Recognition – LSTM

```text
     Webcam Video
           |
    OpenCV Frame Capture
           |
    MediaPipe Holistic
           |
   Landmark Extraction
           |
   1662 Features / Frame
           |
   30-Frame Sequence
           |
       LSTM Model
           |
  40-Class Classification
           |
   Predicted ISL Phrase
```

## 🧠 Model Details

### A. CNN – Static Sign Recognition

The CNN model recognizes static ISL alphabets and digits from images.

**Dataset Information**

| Parameter | Details |
|---|---|
| Dataset | Indian Sign Language Image Dataset |
| Source | Kaggle |
| Number of classes | 35 |
| Classes | A–Z and digits 1–9 |
| Images per class | 1,200 |
| Total images | 42,000 |
| Image dimensions | 128 × 128 RGB |

**Dataset Split**

| Dataset | Images |
|---|---:|
| Training | 33,600 |
| Validation | 4,200 |
| Testing | 4,200 |
| **Total** | **42,000** |

**Model Architecture**

- Convolutional layers with 32, 64, and 128 channels.
- Batch Normalization.
- ReLU activation.
- Max Pooling.
- Adaptive Average Pooling.
- Fully Connected Classification Layer.

**Preprocessing**

- Resize images to 128 × 128.
- Convert images to tensors.
- Normalize RGB channels with mean 0.5 and standard deviation 0.5.

**Evaluation Results**

| Metric | Result |
|---|---:|
| Total test images | 4,200 |
| Correct predictions | 4,200 |
| Test accuracy | 100.00% |

### B. LSTM – Dynamic Phrase Recognition

The LSTM model recognizes dynamic ISL phrases by analyzing sequences of landmark features across multiple frames.

**Dataset Information**

| Parameter | Details |
|---|---|
| Dataset | Common Phrases in Indian Sign Language |
| Source | Mendeley Data |
| Usable classes | 40 |
| Sequences per class | 30 |
| Frames per sequence | 30 |
| Total sequences | 1,200 |
| Total frames | 36,000 |

**Landmark Feature Extraction**

MediaPipe Holistic extracts 1,662 features per frame.

| Landmark Component | Features |
|---|---:|
| Pose | 132 |
| Face | 1,404 |
| Left Hand | 63 |
| Right Hand | 63 |
| **Total** | **1,662** |

**Dataset Split**

| Dataset | Sequences |
|---|---:|
| Training | 960 |
| Validation | 120 |
| Testing | 120 |
| **Total** | **1,200** |

**LSTM Architecture**

| Parameter | Value |
|---|---|
| Input size | 1,662 |
| Sequence length | 30 |
| LSTM layers | 1 |
| Hidden size | 64 |
| Dropout | 0.2 |
| Output classes | 40 |
| Optimizer | Adam |
| Loss function | Cross Entropy Loss |

**Evaluation Results**

| Metric | Result |
|---|---:|
| Total test sequences | 120 |
| Correct predictions | 91 |
| Test accuracy | 75.83% |

*Note: The phrase test set contains only three sequences per class. The reported accuracy is an experimental result on this split and should not be interpreted as guaranteed real-world performance.*

## 📊 Overall Performance

| Model | Recognition Task | Classes | Test Accuracy |
|---|---|---:|---:|
| CNN | Static alphabets and digits | 35 | 100.00% |
| LSTM | Dynamic ISL phrases | 40 | 75.83% |

These results are based on the current held-out test datasets. Performance on unseen signers and real-world recording environments may differ.

## 📂 Project Structure

```text
SignSpeak/
│
├── dataset/                  # Static image dataset (not included)
├── phrase_dataset/           # Prepared phrase features (not included)
│
├── models/
│   ├── sign_model.pth
│   ├── isl_phrase_lstm.pth
│   └── class_names.json
│
├── analyze_confusion.py
├── check_duplicates.py
├── check_images.py
├── collect_data.py
├── evaluate_model.py
├── evaluate_phrase_model.py
├── hand_detection.py
├── inspect_dataset.py
├── live_phrase_recognition.py
├── predict_image.py
├── prepare_dataset.py
├── split_dataset.py
├── train_model.py
├── train_phrase_model.py
├── verify_classes.py
├── visualize_dataset.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10
- Git
- Webcam (for live phrase recognition)
- Windows, Linux, or macOS

### 1. Clone the Repository

```bash
git clone https://github.com/rithikashanmugam-web/SignSpeak.git
cd SignSpeak
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install PyTorch and Torchvision

For CPU-based execution:

```bash
python -m pip install torch==2.14.1 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Other Dependencies

```bash
python -m pip install -r requirements.txt
```

## 📥 Dataset Preparation

The datasets are not included in this repository. Download them from their original sources and follow their respective licensing and usage requirements.

### Static Sign Dataset

Download the [Indian Sign Language Image Dataset from Kaggle](https://www.kaggle.com/datasets/vaishnaviasonawane/indian-sign-language-dataset).

Place and prepare the images according to the folder structure expected by the dataset preparation scripts.

### Dynamic Phrase Dataset

Obtain the Common Phrases in Indian Sign Language dataset from its original Mendeley Data publication.

The source data should follow this structure:

```text
MP_Data/
├── hello/
│   ├── 0/
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   └── 1/
│       └── ...
├── yes/
└── ...
```

Each complete sequence contains 30 frames, with 1,662 features per frame.

## ▶️ Running the Project

### Static Sign Prediction

```bash
python predict_image.py
```

### Hand Detection

```bash
python hand_detection.py
```

### Live Phrase Recognition

```bash
python live_phrase_recognition.py
```

The live phrase recognition module uses the webcam to capture video, extract landmarks, process 30-frame sequences, and predict an ISL phrase.

Press `Q` to close the webcam window.

## 🏋️ Model Training and Evaluation

**Train the CNN model**

```bash
python train_model.py
```

**Evaluate the CNN model**

```bash
python evaluate_model.py
```

**Prepare the phrase dataset**

```bash
python prepare_dataset.py
```

**Train the LSTM model**

```bash
python train_phrase_model.py
```

**Evaluate the LSTM model**

```bash
python evaluate_phrase_model.py
```

## ⚠️ Current Limitations

- The phrase recognition model has a relatively small number of sequences per class.
- The phrase test set contains only three sequences per class.
- Recognition can be affected by lighting, camera positioning, and signer variation.
- The webcam landmark extraction pipeline assumes compatibility with the feature ordering and preprocessing used by the source dataset.
- The current system recognizes predefined classes and does not yet translate continuous signing into complete sentences.

## 🚀 Future Enhancements

- Expand the dataset with more signers and recording conditions.
- Improve recognition accuracy and generalization.
- Implement continuous sign language recognition.
- Convert recognized signs into meaningful sentences.
- Integrate text-to-speech functionality.
- Develop a user-friendly graphical interface.
- Expand the ISL vocabulary and phrase library.
- Explore real-time communication assistance applications.

## 🙌 Acknowledgements

- Indian Sign Language Image Dataset contributors.
- Creators of the Common Phrases in Indian Sign Language dataset.
- PyTorch development community.
- OpenCV development community.
- MediaPipe development team.

## 📄 License

A project license has not yet been selected. Please check the original dataset licenses and redistribution permissions before reusing or redistributing datasets or model artifacts.

---

## 👩‍💻 Author

**SignSpeak – Indian Sign Language Recognition Using Deep Learning**

Developed as an Artificial Intelligence and Machine Learning project.

**GitHub:** [rithikashanmugam-web](https://github.com/rithikashanmugam-web)

---

⭐ If you find this project interesting, consider giving the repository a star!

**SignSpeak – Bridging communication through AI.** 🤟
