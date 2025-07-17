import os
import numpy as np
import random
from multiprocessing import Pool
from functools import partial

def load_img(img_path):
    """Load a single .npy image or mask with error handling."""
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File {img_path} not found")
        data = np.load(img_path)
        return data
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

def load_batch(img_dir, img_list, mask_dir, mask_list, batch_indices):
    """Load a batch of images and masks."""
    batch_images = []
    batch_masks = []

    for idx in batch_indices:
        img_path = os.path.join(img_dir, img_list[idx])
        mask_path = os.path.join(mask_dir, mask_list[idx])

        image = load_img(img_path)
        mask = load_img(mask_path)

        if image is None or mask is None:
            continue  # Skip invalid files

        batch_images.append(image)
        batch_masks.append(mask)

    if not batch_images:  # Handle empty batch
        return None, None

    return np.array(batch_images), np.array(batch_masks)

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size, split_ratio=0.75, mode='train', shuffle=True, num_workers=4):
    """
    Custom data generator for BraTS 2020 dataset with dynamic train/validation split.
    
    Args:
        img_dir (str): Directory containing image .npy files.
        img_list (list): List of image file names.
        mask_dir (str): Directory containing mask .npy files.
        mask_list (list): List of mask file names.
        batch_size (int): Number of samples per batch.
        split_ratio (float): Fraction of data to use for training (e.g., 0.75 for 75% train, 25% validation).
        mode (str): 'train' or 'validation' to select the data subset.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes for parallel loading.
    
    Yields:
        tuple: (batch_images, batch_masks) - NumPy arrays of shape (batch_size, 128, 128, 128, 4) and 
               (batch_size, 128, 128, 128, 4) for images and masks, respectively.
    """
    L = len(img_list)
    if L != len(mask_list):
        raise ValueError("Image and mask lists must have the same length")

    # Create train/validation split
    indices = list(range(L))
    split_idx = int(L * split_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Select indices based on mode
    if mode == 'train':
        active_indices = train_indices
    elif mode == 'validation':
        active_indices = val_indices
    else:
        raise ValueError("Mode must be 'train' or 'validation'")

    L_active = len(active_indices)

    while True:
        if shuffle:
            random.shuffle(active_indices)  # Shuffle indices at the start of each epoch

        batch_start = 0
        while batch_start < L_active:
            batch_end = min(batch_start + batch_size, L_active)
            batch_indices = active_indices[batch_start:batch_end]

            # Load batch in parallel
            with Pool(num_workers) as pool:
                batch_loader = partial(load_batch, img_dir, img_list, mask_dir, mask_list)
                batch_images, batch_masks = batch_loader(batch_indices)

            if batch_images is None or batch_masks is None:
                print("Skipping empty batch")
                batch_start += batch_size
                continue

            yield batch_images, batch_masks

            batch_start += batch_size

# Test the generator
from matplotlib import pyplot as plt

# Define paths
img_dir = "input_data_4channels/images/"
mask_dir = "input_data_4channels/masks/"
img_list = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
mask_list = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

batch_size = 2
split_ratio = 0.75  # 75% train, 25% validation

# Create training generator
train_img_datagen = imageLoader(
    img_dir, img_list, 
    mask_dir, mask_list, 
    batch_size, split_ratio=split_ratio, mode='train', shuffle=True, num_workers=4
)

# Create validation generator
val_img_datagen = imageLoader(
    img_dir, img_list, 
    mask_dir, mask_list, 
    batch_size, split_ratio=split_ratio, mode='validation', shuffle=False, num_workers=4
)

# Fetch and visualize a batch from the training generator
img, msk = next(train_img_datagen)

img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]
test_mask = np.argmax(test_mask, axis=3)  # Convert one-hot to class indices

n_slice = random.randint(0, 127)
plt.figure(figsize=(15, 8))

plt.subplot(231)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1')
plt.subplot(233)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
plt.title('Image t1ce')
plt.subplot(234)
plt.imshow(test_img[:, :, n_slice, 3], cmap='gray')
plt.title('Image t2')
plt.subplot(235)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()