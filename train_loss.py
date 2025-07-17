import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dropout, Conv3DTranspose, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import CosineDecay
from data_gen import imageLoader
from tqdm.keras import TqdmCallback

# Enable mixed precision to reduce memory usage and speed up training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define U-Net Model with L2 Regularization and Increased Dropout
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS), dtype=tf.float32)

    # Contracting Path
    c1 = Conv3D(32, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.3)(c1)  # Increased from 0.2 to 0.3
    c1 = Conv3D(32, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling3D((2,2,2))(c1)

    c2 = Conv3D(64, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.3)(c2)  # Increased from 0.2 to 0.3
    c2 = Conv3D(64, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling3D((2,2,2))(c2)

    c3 = Conv3D(128, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.4)(c3)  # Increased from 0.3 to 0.4
    c3 = Conv3D(128, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling3D((2,2,2))(c3)

    c4 = Conv3D(256, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.4)(c4)  # Increased from 0.3 to 0.4
    c4 = Conv3D(256, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c4)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.4)(c4)  # Increased from 0.3 to 0.4

    # Expanding Path
    u6 = Conv3DTranspose(128, (2,2,2), strides=(2,2,2), padding='same')(c4)
    u6 = concatenate([u6, c3])
    c6 = Conv3D(128, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.4)(c6)  # Increased from 0.3 to 0.4
    c6 = Conv3D(128, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv3DTranspose(64, (2,2,2), strides=(2,2,2), padding='same')(c6)
    u7 = concatenate([u7, c2])
    c7 = Conv3D(64, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.3)(c7)  # Increased from 0.2 to 0.3
    c7 = Conv3D(64, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv3DTranspose(32, (2,2,2), strides=(2,2,2), padding='same')(c7)
    u8 = concatenate([u8, c1])
    c8 = Conv3D(32, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.3)(c8)  # Increased from 0.2 to 0.3
    c8 = Conv3D(32, (3,3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c8)
    c8 = BatchNormalization()(c8)

    outputs = Conv3D(num_classes, (1,1,1), activation='softmax', dtype='float32')(c8)
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

# Loss Functions
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Adjusted class weights: Increased weight for background (class 0) to reduce over-segmentation
    class_weights = tf.convert_to_tensor([2.0, 22.53, 22.53, 26.21], dtype=tf.float32)
    intersection = tf.reduce_sum(class_weights[None, None, None, None, :] * y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(class_weights[None, None, None, None, :] * (y_true + y_pred), axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice, axis=-1)

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
    pt = tf.exp(-ce)
    focal_loss = alpha * tf.pow(1.0 - pt, gamma) * ce
    return tf.reduce_mean(focal_loss, axis=-1)

def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

# New Loss Functions
def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6):
    """
    Tversky Loss: alpha controls false negatives, beta controls false positives.
    Higher beta penalizes false positives more, reducing over-segmentation.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    class_weights = tf.convert_to_tensor([2.0, 22.53, 22.53, 26.21], dtype=tf.float32)
    
    true_pos = tf.reduce_sum(class_weights[None, None, None, None, :] * y_true * y_pred, axis=[1, 2, 3])
    false_neg = tf.reduce_sum(class_weights[None, None, None, None, :] * y_true * (1 - y_pred), axis=[1, 2, 3])
    false_pos = tf.reduce_sum(class_weights[None, None, None, None, :] * (1 - y_true) * y_pred, axis=[1, 2, 3])
    
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return 1.0 - tf.reduce_mean(tversky, axis=-1)

def generalized_dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Generalized Dice Loss: Weights classes inversely proportional to their frequency.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Compute weights as inverse of class frequency (squared)
    class_frequencies = tf.reduce_sum(y_true, axis=[1, 2, 3])
    weights = 1.0 / (tf.square(class_frequencies) + smooth)
    
    intersection = tf.reduce_sum(weights * tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3]), axis=-1)
    union = tf.reduce_sum(weights * tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]), axis=-1)
    
    gdl = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(gdl)

def combo_loss(y_true, y_pred, dice_weight=0.5, ce_weight=0.5, smooth=1e-6):
    """
    Combo Loss: Combines Dice Loss with Weighted Cross-Entropy.
    """
    # Dice Loss
    dice = dice_loss(y_true, y_pred, smooth=smooth)
    
    # Weighted Cross-Entropy
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    class_weights = tf.convert_to_tensor([2.0, 22.53, 22.53, 26.21], dtype=tf.float32)
    ce = tf.reduce_mean(class_weights[None, None, None, None, :] * tf.keras.losses.categorical_crossentropy(y_true, y_pred), axis=[1, 2, 3])
    
    return dice_weight * dice + ce_weight * tf.reduce_mean(ce)

# Metrics
def per_class_dice(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    class_weights = tf.convert_to_tensor([2.0, 22.53, 22.53, 26.21], dtype=tf.float32)
    intersection = tf.reduce_sum(class_weights * y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(class_weights * (y_true + y_pred), axis=[1, 2, 3])
    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)

# Paths
train_img_dir = "input_data_4channels/images/"
train_mask_dir = "input_data_4channels/masks/"

# Verify dataset paths exist to avoid FileNotFoundError
if not os.path.exists(train_img_dir) or not os.path.exists(train_mask_dir):
    raise FileNotFoundError(f"Dataset directories not found: {train_img_dir} or {train_mask_dir}")

img_list = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.npy')])
mask_list = sorted([f for f in os.listdir(train_mask_dir) if f.endswith('.npy')])

# Verify that image and mask lists are not empty
if not img_list or not mask_list:
    raise ValueError("No .npy files found in the dataset directories.")
if len(img_list) != len(mask_list):
    raise ValueError(f"Mismatch between number of images ({len(img_list)}) and masks ({len(mask_list)}).")

print(f"Found {len(img_list)} images, {len(mask_list)} masks")

# Parameters
batch_size = 1
epochs = 50
initial_learning_rate = 1e-3
split_ratio = 0.75

# Data Generators
try:
    train_generator = imageLoader(
        train_img_dir, img_list, train_mask_dir, mask_list, 
        batch_size, split_ratio=split_ratio, mode='train', shuffle=True, num_workers=4
    )
    val_generator = imageLoader(
        train_img_dir, img_list, train_mask_dir, mask_list, 
        batch_size, split_ratio=split_ratio, mode='validation', shuffle=False, num_workers=4
    )
except Exception as e:
    raise RuntimeError(f"Error initializing data generators: {str(e)}")

# Calculate steps
total_samples = len(img_list)
train_samples = int(total_samples * split_ratio)
val_samples = total_samples - train_samples
steps_per_epoch = train_samples // batch_size
val_steps = val_samples // batch_size

print(f"Training: {train_samples} samples, Validation: {val_samples} samples")
print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {val_steps}")

# Test generators to ensure they work correctly
print("Testing train generator...")
try:
    images, masks = next(train_generator)
    print(f"Train batch shape - Images: {images.shape}, Masks: {masks.shape}")
    if images.shape != (batch_size, 128, 128, 128, 4) or masks.shape != (batch_size, 128, 128, 128, 4):
        raise ValueError(f"Unexpected shapes from train generator: Images {images.shape}, Masks {masks.shape}")
except Exception as e:
    raise RuntimeError(f"Error testing train generator: {str(e)}")

print("Testing validation generator...")
try:
    val_images, val_masks = next(val_generator)
    print(f"Validation batch shape - Images: {val_images.shape}, Masks: {val_masks.shape}")
    if val_images.shape != (batch_size, 128, 128, 128, 4) or val_masks.shape != (batch_size, 128, 128, 128, 4):
        raise ValueError(f"Unexpected shapes from validation generator: Images {val_images.shape}, Masks {val_masks.shape}")
except Exception as e:
    raise RuntimeError(f"Error testing validation generator: {str(e)}")

# Convert generators to tf.data.Dataset (without augmentation)
def create_dataset(generator):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, 128, 128, 128, 4], [None, 128, 128, 128, 4])
    )
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_generator)
val_dataset = create_dataset(val_generator)

# Loop over all loss functions
loss_functions = [
    ('dice', dice_loss, 'brats_3d_4channel_dice.h5'),
    ('focal', focal_loss, 'brats_3d_4channel_focal.h5'),
    ('combined', combined_loss, 'brats_3d_4channel_combined.h5'),
    ('tversky', tversky_loss, 'brats_3d_4channel_tversky.h5'),
    ('generalized_dice', generalized_dice_loss, 'brats_3d_4channel_gdl.h5'),
    ('combo', combo_loss, 'brats_3d_4channel_combo.h5')
]

for loss_name, loss_fn, model_name in loss_functions:
    print(f"\nTraining with {loss_name} loss...\n")
    
    # Define Cosine Decay Learning Rate Schedule
    lr_schedule = CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=epochs * steps_per_epoch,
        alpha=1e-6  # Minimum learning rate
    )
    
    # Compile Model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = simple_unet_model(128, 128, 128, 4, 4)
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss=loss_fn,
            metrics=['accuracy', MeanIoU(num_classes=4, name='mean_iou'), per_class_dice]
        )

    # Callbacks
    checkpoint = ModelCheckpoint(f"best_model_4channel_{loss_name}.h5", save_best_only=True, monitor="val_loss", mode="min")
    early_stopping = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    tqdm_callback = TqdmCallback(verbose=2)

    # Train Model
    try:
        history = model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=[checkpoint, early_stopping, tqdm_callback],
            verbose=0
        )
    except Exception as e:
        print(f"Error during training with {loss_name} loss: {str(e)}")
        continue

    # Save Model
    try:
        model.save(model_name)
        print(f"Model saved as {model_name}")
    except Exception as e:
        print(f"Error saving model {model_name}: {str(e)}")
        continue

    # Plot Metrics
    plt.figure(figsize=(12, 10))
    
    # Plot Loss
    plt.subplot(3, 1, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Over Epochs ({loss_name} Loss)')
    plt.legend()
    plt.grid(True)

   
    plt.tight_layout()
    plt.savefig(f"training_metrics_{loss_name}.png")
    plt.close()
    print(f"Training metrics plot saved as 'training_metrics_{loss_name}.png'")