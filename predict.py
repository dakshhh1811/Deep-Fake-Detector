import tensorflow as tf
import numpy as np
import sys
import os

IMG_HEIGHT = 224
IMG_WIDTH = 224
MODEL_FILE = "best_deepfake_model.keras"
class_names = ['Fake', 'Real']

if not os.path.exists(MODEL_FILE):
    print(f"Error: Model file not found at {MODEL_FILE}")
    sys.exit(1)

print(f"Loading model from {MODEL_FILE}...")
model = tf.keras.models.load_model(MODEL_FILE)

if len(sys.argv) < 2:
    print("Error: No image path provided.")
    print("Usage: python predict.py \"path/to/your/image.jpg\"")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    sys.exit(1)

img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = predictions[0][0]

prediction_class = class_names[int(round(score))]
confidence = (1 - score) if score < 0.5 else score

print(f"\n--- Analysis Complete ---")
print(f"File: {os.path.basename(image_path)}")
print(f"Prediction: {prediction_class}")
print(f"Confidence: {confidence * 100:.2f}%")