# Assignment Compliance

- Python: all project scripts are Python.
- Scikit-learn: `train_test_split`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `classification_report`, `confusion_matrix`.
- CNN only: TensorFlow/Keras CNN is the only classifier.
- Dataset: labeled MNIST digits 0–9.
- Unseen test data: official 10,000 MNIST test images.
- Evaluation: accuracy, precision, recall, F1-score, classification report, confusion matrix, training/validation accuracy and loss graphs.
- User image prediction: command-line image path with TensorFlow image decoding and NumPy preprocessing.
- Pillow, Streamlit and OpenCV: not imported or listed in requirements.

Note: Without Streamlit there is no localhost browser upload UI. Run `python app.py "path\to\digit.png"` to predict an uploaded/saved image.
