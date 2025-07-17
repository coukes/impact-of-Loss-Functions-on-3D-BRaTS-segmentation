# impact-of-Loss-Functions-on-3D-BRaTS-segmentation
The project implements a 3D U-Net model with various loss functions to segment brain tumors into four classes: background, necrotic core, edema, and enhancing tumor. The code includes data preprocessing, a custom data generator, model training, and evaluation with ensemble methods for combining predictions.

Table of Contents





Project Overview



Repository Structure



Dataset



Installation



Usage



Kaggle Notebook



Results



License

Project Overview

The goal of this project is to segment brain tumors from multi-modal MRI scans (FLAIR, T1, T1CE, T2) using a 3D U-Net architecture. The pipeline includes:





Preprocessing: Loading and normalizing MRI scans, combining four modalities into a 4-channel input, and cropping to 128x128x128.



Data Generator: A custom generator for efficient batch loading with train/validation splits.



Model Training: Training a 3D U-Net with multiple loss functions and evaluating performance using Dice, IoU, and accuracy metrics.



Ensemble Methods: Combining predictions from multiple models using intersection, majority voting, and weighted averaging.



Visualization: Plotting ground truth and predicted masks for qualitative evaluation.

Repository Structure

BraTS2020-Segmentation/
├── scripts/
│   ├── data_gen.py           # Custom data generator for loading BraTS data
│   ├── predicted_masks.py    # Evaluation and visualization of model predictions
│   ├── pretraitemenet.py     # Data preprocessing for BraTS dataset
│   ├── train_loss.py         # U-Net model definition and training with various loss functions
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore                # Files to ignore in Git
└── LICENSE                   # MIT License

Dataset

The project uses the BraTS 2020 dataset, which contains multi-modal MRI scans and segmentation masks. The dataset is available on Kaggle. The preprocessing script (pretraitemenet.py) converts the data into 4-channel .npy files for efficient loading.

Installation





Clone the repository:

git clone https://github.com/yourusername/BraTS2020-Segmentation.git
cd BraTS2020-Segmentation



Install dependencies:

pip install -r requirements.txt



Download the BraTS 2020 dataset from Kaggle and place it in the BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/ directory.

Usage





Preprocess Data: Run the preprocessing script to convert MRI scans into 4-channel .npy files:

python scripts/pretraitemenet.py

This will create input_data_4channels/images/ and input_data_4channels/masks/ directories.



Train Models: Train the 3D U-Net with different loss functions:

python scripts/train_loss.py

Models will be saved in the current directory with names like brats_3d_4channel_dice.h5.



Evaluate and Visualize: Generate predictions and visualize results:

python scripts/predicted_masks.py

This will produce plots (prediction_sample_methods_*.png) comparing ground truth and predicted masks.

Kaggle Notebook

A Kaggle notebook demonstrating the project is available here. It includes data preprocessing, model training, and evaluation with visualizations.

Results

The project evaluates models using Dice scores, IoU, and accuracy. Ensemble methods (intersection, majority voting, weighted averaging) improve segmentation performance. Example results are visualized in prediction_sample_methods_*.png files.
