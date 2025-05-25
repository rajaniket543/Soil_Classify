# 🌱 Soil Image Classification Challenge 2025 (IIT Ropar, Annam.ai)

This repository contains modular and scalable solutions for two soil image classification tasks:

- **Challenge 1**: Multi-class classification (`Alluvial`, `Black`, `Clay`, `Red`)
- **Challenge 2**: Binary classification (`Soil` vs. `Not-Soil` using CIFAR10 negatives)

---

## 📁 Project Structure

```
.
├── challenge_1/
│   ├── training.ipynb
│   ├── inference.ipynb
│   ├── submission.csv
│   ├── best_model.pth
│   └── src/
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       └── inference.py
├── challenge_2/
│   ├── training.ipynb
│   ├── inference.ipynb
│   ├── submission.csv
│   ├── best_model.pth
│   └── src/
│       ├── dataset.py
│       ├── preprocess.py
│       ├── model.py
│       ├── train.py
│       └── inference.py
├── data/
│   └── soil_competition-2025/
│       ├── train/
│       ├── test/
│       ├── train_labels.csv
│       ├── test_ids.csv
│       ├── train_negatives/
│       ├── negatives_labels.csv
│       └── train_full_labels.csv
└── README.md
```

---

## ✅ Environment Setup

### 1. 🔧 Create and activate virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### 2. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:

```bash
pip install torch torchvision pandas scikit-learn pillow tqdm
```

---

## 📂 Dataset Setup

Place your dataset inside the following directory:

```
data/soil_competition-2025/
├── train/                  # Training images (soil)
├── test/                   # Test images
├── train_labels.csv        # With columns: image_id, soil_type
├── test_ids.csv            # With column: image_id
```

For Challenge 2, CIFAR-10 will be downloaded automatically to generate negative samples.

---

## 🚀 Challenge 1 – Soil Type Classification

### 📍 Navigate to folder

```bash
cd challenge_1
```

### ▶️ Train the Model

Open and run `training.ipynb`:

- Loads soil image dataset
- Trains ResNet/EfficientNet
- Saves `best_model.pth`

### 🧪 Run Inference

Open and run `inference.ipynb`:

- Loads test images
- Runs inference using the best model
- Saves predictions to `submission.csv`

---

## 🚀 Challenge 2 – Soil vs. Not-Soil Binary Classification

### 📍 Navigate to folder

```bash
cd challenge_2
```

### 🛠️ Prepare Data (auto-run)

- CIFAR10 negative images are auto-downloaded and saved under `train_negatives/`
- Merged label CSV is saved as `train_full_labels.csv`

### ▶️ Train the Model

Open and run `training.ipynb`:

- Loads combined soil + negative dataset
- Trains binary classifier
- Saves best model to `best_model.pth`

### 🧪 Run Inference

Open and run `inference.ipynb`:

- Predicts binary labels (soil=1, not-soil=0)
- Outputs `submission.csv` for upload

---

## 🧠 Model Highlights

- Architecture: ResNet18 / ResNet50 / EfficientNet (customizable)
- Loss: CrossEntropy / FocalLoss (optional)
- Metric: F1 Score (used to save best model)
- Optimizer: Adam / SGD
- Transforms: Resize, Normalize, Flip, Rotation (torchvision)

---

## 📊 Tips for Better Performance

- Use `EfficientNet` for stronger performance on small datasets
- Enable `FocalLoss` for handling class imbalance in Challenge 2
- Tune batch size, learning rate, and scheduler settings
- Augment dataset with rotations, crops, color jitter, etc.



## 🙏 Acknowledgements

- Dataset provided by IIT Ropar & Annam.ai
- CIFAR10 negatives sourced from torchvision for binary classification

---

## 📜 License

This project is licensed for academic use only during the Soil Image Classification Challenge 2025.

---
