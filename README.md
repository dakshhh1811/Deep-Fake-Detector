# DeepFake Image Detection

This project is a deep learning system for detecting AI-generated synthetic images (DeepFakes). It uses a Convolutional Neural Network (CNN) based on the **EfficientNetB4** architecture, implemented in TensorFlow and Keras.

The model is trained on a large dataset of real and fake images to achieve high-accuracy binary classification.

## Project Structure

```
DeepFake/
├── best_deepfake_model.keras   # (This is the output after training)
├── df3.py                      # (The main training & evaluation script)
├── img.py                      # (Script to generate the lit. review chart)
├── predict.py                  # (Script to predict a single image)
├── README.md                   # (This file)
└── requirements.txt            # (The dependencies file)
```

## 1. Installation

Follow these steps to set up your environment.

### A. Clone the Repository (Optional)
If your project is in a git repository:
```bash
git clone [https://your-repository-url.git](https://your-repository-url.git)
cd DeepFake
```

### B. Create a Virtual Environment
It is highly recommended to use a virtual environment to avoid conflicts.
```bash
# Create a new environment
python -m venv venv

# Activate the environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### C. Install Dependencies
Install all the required libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

## 2. Dataset Setup

This project **will not run** unless your dataset is set up correctly.

The script `df3.py` uses **hard-coded local paths** (`C:\Users\SARTHAK\OneDrive\Desktop\Dataset\...`). You must either:
1.  Place your dataset in that *exact* location.
2.  **OR** open `df3.py` and change the `TRAIN_DIR`, `VAL_DIR`, and `TEST_DIR` variables to match your dataset's location.

The required folder structure is:
```
C:\Users\SARTHAK\OneDrive\Desktop\Dataset
├── Test
│   ├── Fake
│   └── Real
├── Train
│   ├── Fake
│   └── Real
└── Validation
    ├── Fake
    └── Real
```

## 3. How to Run the Program

There are three main scripts you can run.

### A. Train the Model
This is the main script that trains, validates, and tests the model.

1.  Open your terminal or command prompt.
2.  Make sure your virtual environment is activated.
3.  Run the `df3.py` script:
    ```bash
    python df3.py
    ```
4.  This process **will take a long time**. It will show the progress of each epoch.
5.  When finished, it will save the best-performing model as **`best_deepfake_model.keras`** and print the final **Test Accuracy**.

### B. Predict a Single Image
Once you have the `best_deepfake_model.keras` file, you can use `predict.py` to test a single image.

**Note:** You will need to create the `predict.py` file. Here is the code for it:
```python
# predict.py
import tensorflow as tf
import numpy as np
import sys
import os

# --- Settings ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
MODEL_FILE = "best_deepfake_model.keras"
# The class names must be in the same order as the training script
# 0 = Fake, 1 = Real
class_names = ['Fake', 'Real'] 

# --- 1. Load Model ---
if not os.path.exists(MODEL_FILE):
    print(f"Error: Model file not found at {MODEL_FILE}")
    sys.exit(1)
    
print(f"Loading model from {MODEL_FILE}...")
model = tf.keras.models.load_model(MODEL_FILE)

# --- 2. Get Image Path ---
if len(sys.argv) < 2:
    print("Error: No image path provided.")
    print("Usage: python predict.py \"path/to/your/image.jpg\"")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    sys.exit(1)

# --- 3. Load and Prepare Image ---
try:
    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
except Exception as e:
    print(f"Error loading image: {e}")
    sys.exit(1)

# --- 4. Make Prediction ---
predictions = model.predict(img_array)
score = predictions[0][0] # Get the single score from the batch

# --- 5. Show Result ---
prediction_class = class_names[int(round(score))]
confidence = (1 - score) if score < 0.5 else score

print(f"\n--- Analysis Complete ---")
print(f"File: {os.path.basename(image_path)}")
print(f"Prediction: {prediction_class}")
print(f"Confidence: {confidence * 100:.2f}%")
print(f"(Raw Score: {score:.4f})")
```

**To run the prediction script:**
```bash
# Pass the path to your image in quotes
python predict.py "C:\Users\SARTHAK\Downloads\my_test_photo.jpg"
```
**Example Output:**
```
Loading model from best_deepfake_model.keras...
--- Analysis Complete ---
File: my_test_photo.jpg
Prediction: Real
Confidence: 99.87%
(Raw Score: 0.9987)
```

### C. (Optional) Generate Literature Review Chart
This script generates the bar chart for the research paper.
```bash
python img.py
```
This will create a file named **`Fig_1_Accuracy_Comparison.jpg`** in your project folder.
