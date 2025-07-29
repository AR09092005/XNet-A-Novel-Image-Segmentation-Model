# XNet-A-Novel-Image-Segmentation-Model
X-Net is a novel Image Segmentation model based on a modified version of U-Net. In this project it is used to Segment lung CT Scans.

This project implements an image segmentation model, XNet (a variant of U-Net), to segment features in medical images. The model is trained on a custom dataset containing both "Normal" and "Viral Pneumonia" chest X-ray images and their corresponding masks.

## Project Description

The goal of this project is to train a deep learning model to perform semantic segmentation on medical images. Specifically, the model learns to identify and segment regions of interest (e.g., abnormalities) within chest X-rays.

## Dataset

The project uses two datasets:
- `Normal.zip`: Contains chest X-ray images and masks for normal cases.
- `ViralPneumonia.zip`: Contains chest X-ray images and masks for cases with viral pneumonia.

The notebook assumes these zip files are located in your Google Drive and will be extracted to `/content/data` and `/content/ViralPneumoniaTest` respectively, creating the following structure:

/content/
├── data/
│   └── Normal/
│       ├── images/
│       └── masks/
└── ViralPneumoniaTest/
    └── Viral Pneumonia/
        ├── images/
        └── masks/
## Setup and Installation

1.  **Mount Google Drive:** The notebook requires access to your Google Drive to load the datasets.
2.  **Dependencies:** Install the required Python libraries.
    ```bash
    pip install torch torchvision matplotlib seaborn pandas scikit-learn Pillow
    ```
3.  **Run the Notebook:** Execute the cells in the notebook sequentially.

## Notebook Structure

-   **Data Loading and Preprocessing:**
    -   Mounts Google Drive.
    -   Extracts the "Normal" and "ViralPneumonia" datasets.
    -   Defines a custom `SegmentationDataset` class to load and transform the images and masks.

-   **Model Definition:**
    -   Defines the XNet model architecture, which is a U-Net-like architecture for segmentation.

-   **Training:**
    -   Sets up the training loop with the Adam optimizer and BCEWithLogitsLoss.
    -   Trains the model for a specified number of epochs.
    -   Saves the best model based on validation loss.

-   **Evaluation:**
    -   Loads the trained model.
    -   Evaluates the model on the test set (Viral Pneumonia data).
    -   Calculates and prints metrics: Dice coefficient, IoU (Jaccard index), and ROC AUC.

-   **Visualization:**
    -   Visualizes the model's predictions against the ground truth masks.
    -   Plots the distribution of Dice and Jaccard scores.
    -   Displays the ROC curve.

## Model Architecture: XNet

The XNet model is a convolutional neural network with an encoder-decoder structure, similar to U-Net. It uses convolutional blocks to extract features at different scales and then upsamples these features to generate a segmentation map.

## Training Process

-   **Loss Function:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) is used to measure the difference between the predicted and ground truth masks.
-   **Optimizer:** The Adam optimizer is used to update the model's weights during training.
-   **Epochs:** The model is trained for a specified number of epochs, and the best model is saved based on the validation loss.

## Evaluation Metrics

-   **Dice Coefficient:** Measures the overlap between the predicted and ground truth masks.
-   **Intersection over Union (IoU) / Jaccard Index:** Another metric for measuring the overlap between the predicted and ground truth masks.
-   **ROC AUC:** Evaluates the model's ability to distinguish between positive and negative pixels.

## Visualizing Predictions

The notebook includes code to visualize the input image, the ground truth mask, and the predicted mask side-by-side, providing a qualitative assessment of the model's performance.
