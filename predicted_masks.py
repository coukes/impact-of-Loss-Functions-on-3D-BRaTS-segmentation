import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_gen import imageLoader
import scipy.ndimage as ndimage

# Define Metrics
def per_class_dice(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    class_weights = tf.convert_to_tensor([1.0, 22.53, 22.53, 26.21], dtype=tf.float32)
    intersection = tf.reduce_sum(class_weights * y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(class_weights * (y_true + y_pred), axis=[1, 2, 3])
    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)

# --- IoU Score ---
MASK_CLASSES = 4  # Number of classes (background, necrotic core, edema, enhancing tumor)
def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    iou_total = 0
    for c in range(MASK_CLASSES):
        intersection = np.logical_and(y_true == c, y_pred == c).sum()
        union = np.logical_or(y_true == c, y_pred == c).sum()
        iou = (intersection + smooth) / (union + smooth)
        iou_total += iou
    return iou_total / MASK_CLASSES

# --- Accuracy Score ---
def accuracy_score(y_true, y_pred):
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    correct = np.equal(y_true, y_pred).sum()
    total = y_true.size
    return correct / total

# Post-processing to remove small regions
def post_process_predictions(predictions, min_size=50):
    predictions = np.argmax(predictions, axis=-1)  # Convert softmax to class labels
    processed = np.zeros_like(predictions)
    for class_label in range(1, 4):  # Classes 1, 2, 3 (excluding background)
        binary = (predictions == class_label).astype(np.uint8)
        labeled, num_features = ndimage.label(binary)
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < min_size:
                binary[labeled == i] = 0
        processed[binary == 1] = class_label
    # Reconstruct one-hot encoded format
    one_hot = np.zeros(predictions.shape + (4,), dtype=np.float32)
    for class_label in range(4):
        one_hot[..., class_label] = (processed == class_label).astype(np.float32)
    return one_hot

# Combine masks using intersection method
def combine_masks_intersection(predictions_dict):
    pred_labels = {name: np.argmax(pred, axis=-1) for name, pred in predictions_dict.items()}
    combined_labels = np.zeros_like(list(pred_labels.values())[0], dtype=np.uint8)
    for b in range(combined_labels.shape[0]):
        for x in range(combined_labels.shape[1]):
            for y in range(combined_labels.shape[2]):
                for z in range(combined_labels.shape[3]):
                    labels = [pred_labels[name][b, x, y, z] for name in pred_labels]
                    if all(label == labels[0] for label in labels):
                        combined_labels[b, x, y, z] = labels[0]
                    else:
                        combined_labels[b, x, y, z] = 0  # Background
    combined_one_hot = np.zeros(list(predictions_dict.values())[0].shape, dtype=np.float32)
    for class_label in range(4):
        combined_one_hot[..., class_label] = (combined_labels == class_label).astype(np.float32)
    return combined_one_hot

# Combine masks using majority voting
def combine_masks_majority_voting(predictions_dict):
    pred_labels = {name: np.argmax(pred, axis=-1) for name, pred in predictions_dict.items()}
    combined_labels = np.zeros_like(list(pred_labels.values())[0], dtype=np.uint8)
    for b in range(combined_labels.shape[0]):
        for x in range(combined_labels.shape[1]):
            for y in range(combined_labels.shape[2]):
                for z in range(combined_labels.shape[3]):
                    labels = [pred_labels[name][b, x, y, z] for name in pred_labels]
                    class_counts = np.bincount(labels, minlength=4)
                    max_count = np.max(class_counts)
                    if max_count >= len(predictions_dict) // 2 + 1:  # Majority (at least 3 out of 5 agree)
                        combined_labels[b, x, y, z] = np.argmax(class_counts)
                    else:
                        combined_labels[b, x, y, z] = 0  # Background
    combined_one_hot = np.zeros(list(predictions_dict.values())[0].shape, dtype=np.float32)
    for class_label in range(4):
        combined_one_hot[..., class_label] = (combined_labels == class_label).astype(np.float32)
    return combined_one_hot

# Combine masks using weighted averaging of probabilities
def combine_masks_weighted_average(predictions_dict, weights_dict):
    weights = np.array(list(weights_dict.values())) / np.sum(list(weights_dict.values()))  # Normalize weights
    combined_probs = np.zeros_like(list(predictions_dict.values())[0])
    for idx, (name, pred) in enumerate(predictions_dict.items()):
        combined_probs += weights[idx] * pred
    combined_labels = np.argmax(combined_probs, axis=-1)
    combined_one_hot = np.zeros_like(combined_probs, dtype=np.float32)
    for class_label in range(4):
        combined_one_hot[..., class_label] = (combined_labels == class_label).astype(np.float32)
    return combined_one_hot

# Paths
train_img_dir = "input_data_4channels/images/"
train_mask_dir = "input_data_4channels/masks/"

# Verify dataset paths
if not os.path.exists(train_img_dir) or not os.path.exists(train_mask_dir):
    raise FileNotFoundError(f"Dataset directories not found: {train_img_dir} or {train_mask_dir}")

img_list = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.npy')])
mask_list = sorted([f for f in os.listdir(train_mask_dir) if f.endswith('.npy')])

if len(img_list) != len(mask_list):
    raise ValueError(f"Mismatch between images ({len(img_list)}) and masks ({len(mask_list)}).")

print(f"Found {len(img_list)} images, {len(mask_list)} masks")

# Parameters
batch_size = 5
split_ratio = 0.75

# Data Generator for Validation
val_generator = imageLoader(
    train_img_dir, img_list, train_mask_dir, mask_list, 
    batch_size, split_ratio=split_ratio, mode='validation', shuffle=False, num_workers=4
)

# Calculate validation samples
total_samples = len(img_list)
val_samples = total_samples - int(total_samples * split_ratio)
val_steps = val_samples // batch_size

# Load Models
models = {
    'Dice Loss': load_model('50epoch/brats_3d_4channel_dice.h5', custom_objects={'dice_loss': lambda y_true, y_pred: y_true, 'per_class_dice': per_class_dice}),
    'Combined Loss': load_model('50epoch/brats_3d_4channel_combined.h5', custom_objects={'combined_loss': lambda y_true, y_pred: y_true, 'per_class_dice': per_class_dice}),
    'Tversky Loss': load_model('50epoch/brats_3d_4channel_tversky.h5', custom_objects={'tversky_loss': lambda y_true, y_pred: y_true, 'per_class_dice': per_class_dice}),
    'Focal Loss': load_model('50epoch/brats_3d_4channel_focal.h5', custom_objects={'focal_loss': lambda y_true, y_pred: y_true, 'per_class_dice': per_class_dice}),
    'GEN Dice Loss': load_model('50epoch/brats_3d_4channel_gdl.h5', custom_objects={'generalized_dice_loss': lambda y_true, y_pred: y_true, 'per_class_dice': per_class_dice})
}

# Update weights for weighted averaging based on hypothetical IoU scores
weights = {
    'Dice Loss': 0.62,
    'Combined Loss': 0.59,
    'Tversky Loss': 0.61,
    'Focal Loss': 0.60,
    'GEN Dice Loss': 0.61
}

# Select 5 random indices for visualization
np.random.seed(42)  # For reproducibility
visualization_indices = np.random.choice(val_samples, 5, replace=False)
visualization_indices = sorted(visualization_indices)

# Collect 5 samples for visualization by iterating over the generator
visualization_samples = []
current_idx = 0
vis_idx = 0
for images, masks in val_generator:
    if current_idx == visualization_indices[vis_idx]:
        visualization_samples.append((images, masks))
        vis_idx += 1
        if vis_idx == len(visualization_indices):
            break
    current_idx += 1

# Plot Predictions for 5 Samples (Including Remaining Combination Methods)
class_names = ['Background', 'Necrotic Core', 'Edema', 'Enhancing Tumor']
slice_idx = 64  # Middle slice for visualization

for i, (images, masks) in enumerate(visualization_samples):
    plt.figure(figsize=(40, 5))
    
    # Ground Truth
    plt.subplot(1, 9, 1)
    gt_slice = np.argmax(masks[0, :, :, slice_idx, :], axis=-1)
    plt.imshow(gt_slice, vmin=0, vmax=3)
    plt.title(f"Ground Truth - Sample {i+1}")
    plt.colorbar(label='Class')
    plt.axis('off')
    
    # Individual Model Predictions
    predictions_dict = {}
    for idx, (model_name, model) in enumerate(models.items(), start=2):
        pred = model.predict(images, verbose=0)
        pred = post_process_predictions(pred)
        predictions_dict[model_name] = pred
        dice_score = tf.reduce_mean(per_class_dice(masks, pred)).numpy().item()
        plt.subplot(1, 9, idx)
        pred_slice = np.argmax(pred[0, :, :, slice_idx, :], axis=-1)
        plt.imshow(pred_slice, vmin=0, vmax=3)
        plt.title(f"{model_name} Prediction\nDice = {dice_score:.4f}")
        plt.colorbar(label='Class')
        plt.axis('off')
    
    # Intersection Combined Mask
    intersection_mask = combine_masks_intersection(predictions_dict)
    dice_intersection = tf.reduce_mean(per_class_dice(masks, intersection_mask)).numpy().item()
    plt.subplot(1, 9, 7)
    intersection_slice = np.argmax(intersection_mask[0, :, :, slice_idx, :], axis=-1)
    plt.imshow(intersection_slice, vmin=0, vmax=3)
    plt.title(f"Intersection Combined\nDice = {dice_intersection:.4f}")
    plt.colorbar(label='Class')
    plt.axis('off')
    
    # Majority Voting Combined Mask
    majority_mask = combine_masks_majority_voting(predictions_dict)
    dice_majority = tf.reduce_mean(per_class_dice(masks, majority_mask)).numpy().item()
    plt.subplot(1, 9, 8)
    majority_slice = np.argmax(majority_mask[0, :, :, slice_idx, :], axis=-1)
    plt.imshow(majority_slice, vmin=0, vmax=3)
    plt.title(f"Majority Voting Combined\nDice = {dice_majority:.4f}")
    plt.colorbar(label='Class')
    plt.axis('off')
    
    # Weighted Average Combined Mask
    weighted_mask = combine_masks_weighted_average(predictions_dict, weights)
    dice_weighted = tf.reduce_mean(per_class_dice(masks, weighted_mask)).numpy().item()
    plt.subplot(1, 9, 9)
    weighted_slice = np.argmax(weighted_mask[0, :, :, slice_idx, :], axis=-1)
    plt.imshow(weighted_slice, vmin=0, vmax=3)
    plt.title(f"Weighted Average Combined\nDice = {dice_weighted:.4f}")
    plt.colorbar(label='Class')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"prediction_sample_methods_{i+1}.png")
    plt.close()
    print(f"Saved prediction plot for sample {i+1} as 'prediction_sample_methods_{i+1}.png'")

