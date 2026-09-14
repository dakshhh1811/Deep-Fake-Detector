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

Place a `Dataset` folder in the same directory as `df3.py`, structured as:

Dataset/
├── Test
│   ├── Fake
│   └── Real
├── Train
│   ├── Fake
│   └── Real
└── Validation
    ├── Fake
    └── Real

Paths are resolved automatically relative to the script location — no manual editing needed.

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

Run it directly with an image path:

**To run the prediction script:**
```bash
# Pass the path to your image in quotes
python predict.py "path/to/your/image.jpg"
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
