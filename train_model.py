"""
Train a CNN-only handwritten digit classifier on MNIST.
Scikit-learn is used for a reproducible stratified train/validation split
and evaluation metrics. It is NOT used as a classifier.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
EVAL_DIR = ROOT / "evaluation"
MODEL_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "digit_cnn.keras"
SEED = 42

def build_cnn():
    """The only prediction model in this project: a Convolutional Neural Network."""
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.40),
        keras.layers.Dense(10, activation="softmax")
    ], name="CNN_Digit_Classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    tf.keras.utils.set_random_seed(SEED)

    # MNIST: 60,000 labeled training examples and 10,000 official test examples.
    (x_all, y_all), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scikit-learn performs the stratified train/validation split.
    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=0.10, random_state=SEED, stratify=y_all
    )

    # CNN input: grayscale 28x28 images, normalized to [0, 1].
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_val = (x_val.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    model = build_cnn()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate once on the untouched official test set.
    test_probabilities = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = np.argmax(test_probabilities, axis=1)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "test_samples": int(len(y_test)),
        "classifier": "Convolutional Neural Network (CNN)"
    }
    (EVAL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    report = classification_report(y_test, y_pred, digits=4)
    (EVAL_DIR / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("CNN - MNIST Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    epochs = range(1, len(history.history["accuracy"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history.history["accuracy"], label="Training accuracy")
    ax.plot(epochs, history.history["val_accuracy"], label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("CNN Training and Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "accuracy_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history.history["loss"], label="Training loss")
    ax.plot(epochs, history.history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CNN Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "loss_curve.png", dpi=160)
    plt.close(fig)

    print("\nTEST METRICS")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nClassification report saved to evaluation/classification_report.txt")
    print(f"Trained CNN saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
