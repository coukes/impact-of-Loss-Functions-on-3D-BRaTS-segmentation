# BraTS2020-Segmentation

This repository contains code for brain tumor segmentation using the BraTS 2020 dataset. The project implements a 3D U-Net model with various loss functions (Dice, Focal, Tversky, Generalized Dice, Combo) to segment brain tumors into four classes: background, necrotic core, edema, and enhancing tumor. The code includes data preprocessing, a custom data generator, model training, and evaluation with ensemble methods for combining predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Usage](#usage)
- [Kaggle Notebook](#kaggle-notebook)
- [Results](#results)
- [License](#license)

## Project Overview
The goal of this project is to segment brain tumors from multi-modal MRI scans (FLAIR, T1, T1CE, T2) using a 3D U-Net architecture. The pipeline includes:
- **Preprocessing**: Loading and normalizing MRI scans, combining four modalities into a 4-channel input, and cropping to 128x128x128.
- **Data Generator**: A custom generator for efficient batch loading with train/validation splits.
- **Model Training**: Training a 3D U-Net with multiple loss functions and evaluating performance using Dice, IoU, and accuracy metrics.
- **Ensemble Methods**: Combining predictions from multiple models using intersection, majority voting, and weighted averaging.
- **Visualization**: Plotting ground truth and predicted masks for qualitative evaluation.


## Dataset
The project uses the BraTS 2020 dataset, which contains multi-modal MRI scans and segmentation masks. The dataset is available on [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation). The preprocessing script (`pretraitemenet.py`) converts the data into 4-channel `.npy` files for efficient loading.
- **Install Dependencies**: Install the required Python packages listed in requirements.txt:
- pip install -r requirements.txt

## Usage
