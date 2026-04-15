import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
import os
import matplotlib.pyplot as plt

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. DEFINE ALL PATHS (Using your provided paths) ---
# Using raw strings (r"...") to avoid errors with backslashes

TRAIN_DIR = r"C:\Users\SARTHAK\OneDrive\Desktop\Dataset\Train"
VAL_DIR = r"C:\Users\SARTHAK\OneDrive\Desktop\Dataset\Validation"
TEST_DIR = r"C:\Users\SARTHAK\OneDrive\Desktop\Dataset\Test"

# --- 2. SET MODEL PARAMETERS ---
# From the paper, EfficientNetB4 works well with 224x224
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 16  # You can adjust this based on your GPU memory
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# --- 3. LOAD DATASET ---
# Use image_dataset_from_directory to load the data efficiently

print("Loading datasets...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',       # Labels from 'Real' and 'Fake' folders
    label_mode='binary',     # 0 for 'Fake', 1 for 'Real'
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Show the class names it found (Should be ['Fake', 'Real'])
print(f"Class names: {train_dataset.class_names}")

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# --- 4. BUILD THE MODEL (EfficientNetB4) ---

print("Building the model...")

# Create the Data Augmentation layer (as mentioned in the paper)
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# Load the EfficientNetB4 base model (pre-trained on ImageNet)
# include_top=False removes the final 1000-neuron classifier
base_model = EfficientNetB4(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Freeze the base model
# This prevents the pre-trained weights from being updated
# during the initial training.
base_model.trainable = False

# Create the full model
model = models.Sequential([
    # Input layer
    layers.Input(shape=IMG_SHAPE),
    
    # Data Augmentation
    data_augmentation,
    
    # Rescale pixels from [0, 255] to [0, 1] 
    # (EfficientNet expects this)
    layers.Rescaling(1./255),
    
    # The pre-trained base model
    base_model,
    
    # Flatten the features to a 1D vector
    layers.GlobalAveragePooling2D(),
    
    # Our new classifier head
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Dropout to reduce overfitting
    
    # Output layer: 1 neuron with sigmoid for binary (0/1)
    layers.Dense(1, activation='sigmoid')
], name="DeepFake_Detector_EfficientNetB4")


# Print a summary of the model
model.summary()

# --- 5. COMPILE THE MODEL ---

print("\n--- Compiling the model ---")

model.compile(
    # Adam is a great all-around optimizer
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    
    # BinaryCrossentropy is the correct loss for 0/1 classification
    loss='binary_crossentropy', 
    
    # We want to track accuracy
    metrics=['accuracy'] 
)

# --- 6. DEFINE CALLBACKS ---
# Callbacks are tools that help us train better

# 1. ModelCheckpoint: Save only the *best* model
# It will monitor the validation accuracy and save the model
# only when it improves.
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_deepfake_model.keras",  # File to save the model
    save_best_only=True,
    monitor="val_accuracy",
    mode="max"
)

# 2. EarlyStopping: Stop training if the model isn't improving
# This saves time and prevents overfitting.
# It will monitor validation loss. If it doesn't improve
# for 3 epochs in a row, training will stop.
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3, # Number of epochs to wait for improvement
    restore_best_weights=True # Restore weights from the best epoch
)


# --- 7. TRAIN THE MODEL ---

print("--- Starting Model Training (This will take a while...) ---")

# We'll start with 15 epochs. EarlyStopping will stop it if 
# it finishes sooner.
EPOCHS = 15

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[checkpoint_cb, early_stopping_cb] # Add our callbacks
)

print("--- Model Training Finished ---")

# --- 8. EVALUATE ON THE TEST SET ---

print("\n--- Evaluating Model on the Unseen Test Data ---")

# Load the best model we saved during training
print("Loading best model from 'best_deepfake_model.keras'...")
best_model = tf.keras.models.load_model("best_deepfake_model.keras")

# Evaluate its performance
loss, accuracy = best_model.evaluate(test_dataset)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

import tensorflow as tf
import numpy as np
import sys

# Load the saved model
model = tf.keras.models.load_model("best_deepfake_model.keras")
IMG_HEIGHT = 224
IMG_WIDTH = 224
class_names = ['Fake', 'Real'] # Must match training

# Get image path from command line
image_path = sys.argv[1] 

# Load and prepare the image
img = tf.keras.utils.load_img(
    image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# Make prediction
predictions = model.predict(img_array)
score = predictions[0][0] # Get the single score from the batch

# Print the result
print(f"--- Analyzing image: {image_path} ---")
if score < 0.5:
    print(f"Prediction: FAKE (Confidence: {(1 - score) * 100:.2f}%)")
else:
    print(f"Prediction: REAL (Confidence: {score * 100:.2f}%)")