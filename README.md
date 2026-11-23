# Handwritten Digit Recognition using Python, Scikit-learn and CNN

## This version removes
- Pillow / PIL
- Streamlit
- OpenCV

Image loading and decoding use TensorFlow (`tf.io.decode_image`), and preprocessing uses TensorFlow + NumPy. The application is command-line based, not a browser upload interface.

## Technologies
- Python
- TensorFlow/Keras: CNN model training and prediction
- Scikit-learn: stratified train/validation split and evaluation metrics only
- NumPy: image and numerical processing
- Matplotlib: training curves and confusion matrix

**CNN is the only classifier.** No SVM, KNN, Random Forest, Decision Tree or other classification model is used.

## Requirements
Recommended: Python 3.11 (64-bit).

## Install (Windows PowerShell)
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is not recognized, install 64-bit Python 3.11 first and reopen PowerShell.

## Train the CNN
```powershell
python train_model.py
```
The first run downloads MNIST. It trains a CNN and evaluates it on the untouched official MNIST test set (10,000 images). Training creates:
- `model/digit_cnn.keras`
- `evaluation/metrics.json`
- `evaluation/classification_report.txt`
- `evaluation/confusion_matrix.png`
- `evaluation/accuracy_curve.png`
- `evaluation/loss_curve.png`

## Predict a digit from an image
Place a single-digit image in the project folder or provide its full path:
```powershell
python app.py "C:\Users\Vedant\Downloads\my_digit.png"
```
Supported formats are PNG, JPG/JPEG and BMP, decoded by TensorFlow. The program prints the predicted digit and per-class probabilities. It also saves the processed 28x28 array as `.npy`.

## Assignment checklist
1. Recognize handwritten digits 0-9: yes.
2. Labeled dataset: MNIST.
3. CNN classification: yes, TensorFlow/Keras CNN only.
4. Unseen test data: official MNIST test split.
5. Python + Scikit-learn: Scikit-learn split and metrics.
6. Evaluation: accuracy, precision, recall, F1, classification report, confusion matrix and curves.
7. Image input and prediction: command-line image path, no web interface.

## Important technical clarification
Scikit-learn does not include a native CNN implementation. TensorFlow/Keras implements the CNN. Scikit-learn is used for splitting and evaluation. Removing Streamlit means there is no localhost/browser upload page; prediction is done by supplying an image path in the terminal.
