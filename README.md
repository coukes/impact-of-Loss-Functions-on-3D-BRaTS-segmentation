# BraTS2020-Segmentation

This repository contains code for brain tumor segmentation using the BraTS 2020 dataset. The project implements a 3D U-Net model with various loss functions (Dice, Focal, Tversky, Generalized Dice, Combo) to segment brain tumors into four classes: background, necrotic core, edema, and enhancing tumor. The code includes data preprocessing, a custom data generator, model training, and evaluation with ensemble methods for combining predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Overview
The goal of this project is to segment brain tumors from multi-modal MRI scans (FLAIR, T1, T1CE, T2) using a 3D U-Net architecture. The pipeline includes:
- **Preprocessing**: Loading and normalizing MRI scans, combining four modalities into a 4-channel input, and cropping to 128x128x128.
- **Data Generator**: A custom generator for efficient batch loading with train/validation splits.
- **Model Training**: Training a 3D U-Net with multiple loss functions and evaluating performance using Dice, IoU, and accuracy metrics.
- **Ensemble Methods**: Combining predictions from multiple models using intersection, majority voting, and weighted averaging.
- **Visualization**: Plotting ground truth and predicted masks for qualitative evaluation.

## Repository Structure
```
BraTS2020-Segmentation/

├── data_gen.py           # Custom data generator for loading BraTS data
├── predicted_masks.py    # Evaluation and visualization of model predictions
├── pretraitemenet.py     # Data preprocessing for BraTS dataset
├── train_loss.py         # U-Net model definition and training with various loss functions
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore                # Files to ignore in Git
└── LICENSE                   # MIT License
```

## Dataset
The project uses the BraTS 2020 dataset, which contains multi-modal MRI scans and segmentation masks. The dataset is available on [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation). The preprocessing script (`pretraitemenet.py`) converts the data into 4-channel `.npy` files for efficient loading.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BraTS2020-Segmentation.git
   cd BraTS2020-Segmentation
   ```
2. Install dependencies:
   ```bash
   pip install numpy==1.24.3 tensorflow==2.17.0 nibabel==5.1.0 matplotlib==3.7.2 scikit-learn==1.3.0 scipy==1.11.1 tqdm==4.66.1 tifffile==2023.7.10
   ```
3. Download the BraTS 2020 dataset from Kaggle and place it in the `BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/` directory.
4. **Preprocess Data**:
   Run the preprocessing script to convert MRI scans into 4-channel `.npy` files:
   ```bash
   python pretraitemenet.py
   ```
   This will create `input_data_4channels/images/` and `input_data_4channels/masks/` directories.
5. **Set Up Data Generator**:
   Use the custom data generator (`data_gen.py`) to load batches of preprocessed images and masks efficiently. Configure it with a batch size and train/validation split (e.g., 75% train, 25% validation).
6. **Train Models**:
   Train the 3D U-Net with different loss functions:
   ```bash
   python train_loss.py
   ```
   Models will be saved in the current directory with names like `brats_3d_4channel_dice.h5`.
7. **Evaluate and Visualize**:
   Generate predictions and visualize results:
   ```bash
   python predicted_masks.py
   ```
   This will produce plots (`prediction_sample_methods_*.png`) comparing ground truth and predicted masks.



## Results
The project evaluates models using Dice scores, IoU, and accuracy. Ensemble methods (intersection, majority voting, weighted averaging) improve segmentation performance. Example results are visualized in `prediction_sample_methods_*.png` files, showing comparisons between ground truth and predicted masks for five validation samples.

***Performance was evaluated using the Dice Similarity Coefficient, Intersection-Over-Union (IoU), and accuracy metrics. The results demonstrated strong segmentation accuracy, with ensemble methods, particularly majority voting and weighted average, outperforming individual models. High Dice scores for enhancing tumor and edema regions underscored the framework’s ability to capture intricate 3D spatial features. Visualizations of ground truth and predicted masks for five validation samples confirmed the models’ precision and the effectiveness of ensemble approaches.***

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
