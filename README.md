# XNet-A-Novel-Image-Segmentation-Model

X-Net is a novel image segmentation model based on a modified version of U-Net. In this project, it is used to segment lung CT scans.

This project implements an image segmentation model, **XNet** (a variant of U-Net), to segment features in medical images. The model is trained on a custom dataset containing both **Normal** and **Viral Pneumonia** chest X-ray images along with their corresponding segmentation masks.

---

## Project Description

The goal of this project is to train a deep learning model to perform semantic segmentation on medical images. Specifically, the model learns to identify and segment regions of interest (e.g., abnormalities) within chest X-ray images.

---

## Dataset

The project uses two datasets:

- `Normal.zip` – Contains chest X-ray images and masks for normal cases.
- `ViralPneumonia.zip` – Contains chest X-ray images and masks for viral pneumonia cases.

The notebook assumes these ZIP files are stored in your Google Drive and will be extracted to the following locations:

```text
/content/
├── data/
│   └── Normal/
│       ├── images/
│       └── masks/
└── ViralPneumoniaTest/
    └── Viral Pneumonia/
        ├── images/
        └── masks/
```

---

## Repository Structure

```text
XNet-A-Novel-Image-Segmentation-Model/
├── .gitignore
├── README.md
└── Xnetimp.ipynb
```

---

## Setup and Installation

### 1. Mount Google Drive

The notebook requires access to your Google Drive to load the datasets.

### 2. Install Dependencies

```bash
pip install torch torchvision matplotlib seaborn pandas scikit-learn Pillow
```

### 3. Run the Notebook

Execute the notebook cells sequentially.

---

## Notebook Structure

### Data Loading and Preprocessing

- Mounts Google Drive.
- Extracts the **Normal** and **Viral Pneumonia** datasets.
- Defines a custom `SegmentationDataset` class for loading and transforming images and masks.

### Model Definition

- Implements the **XNet** architecture, a modified U-Net for semantic segmentation.

### Training

- Uses the Adam optimizer and `BCEWithLogitsLoss`.
- Trains the model for the specified number of epochs.
- Saves the best-performing model based on validation loss.

### Evaluation

- Loads the trained model.
- Evaluates performance on the Viral Pneumonia test dataset.
- Computes:
  - Dice Coefficient
  - Intersection over Union (IoU / Jaccard Index)
  - ROC AUC

### Visualization

- Displays the input image, ground truth mask, and predicted mask.
- Plots the distribution of Dice and IoU scores.
- Displays the ROC curve.

---

## Model Architecture: XNet

The XNet model is a convolutional neural network with an encoder-decoder architecture inspired by U-Net. It extracts hierarchical image features through convolutional blocks and reconstructs high-resolution segmentation maps using an upsampling decoder with skip connections.

---

## Training Process

- **Loss Function:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)
- **Optimizer:** Adam
- **Training:** The model is trained for a specified number of epochs, and the best-performing weights are saved based on validation loss.

---

## Evaluation Metrics

- **Dice Coefficient:** Measures the overlap between predicted and ground truth masks.
- **Intersection over Union (IoU) / Jaccard Index:** Measures segmentation quality by comparing prediction overlap.
- **ROC AUC:** Evaluates the model's ability to distinguish between positive and negative pixels.

---

## Visualizing Predictions

The notebook includes visualization utilities that display:

- Original chest X-ray
- Ground truth segmentation mask
- Predicted segmentation mask

These qualitative visualizations complement the quantitative evaluation metrics.
