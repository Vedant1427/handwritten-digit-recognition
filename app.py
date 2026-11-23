"""Command-line interface for CNN handwritten digit prediction."""
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
from image_processing import preprocess_image_path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "digit_cnn.keras"

def main():
    parser = argparse.ArgumentParser(
        description="Predict one handwritten digit (0-9) using the trained CNN."
    )
    parser.add_argument("image", help="Path to a PNG, JPG, JPEG or BMP image containing one digit")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(
            "Trained CNN model not found. First run: python train_model.py"
        )

    try:
        model_input, processed = preprocess_image_path(args.image)
        model = tf.keras.models.load_model(MODEL_PATH)
        probabilities = model.predict(model_input, verbose=0)[0]
        predicted = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted]) * 100

        print("\n--- CNN Digit Recognition Result ---")
        print(f"Image: {args.image}")
        print(f"Predicted digit: {predicted}")
        print(f"Confidence: {confidence:.2f}%")
        print("\nClass probabilities:")
        for digit, probability in enumerate(probabilities):
            print(f"{digit}: {probability*100:.2f}%")

        # Save the 28x28 preprocessed image as a NumPy array for inspection.
        output_path = Path(args.image).with_name(Path(args.image).stem + "_28x28.npy")
        np.save(output_path, processed)
        print(f"\nPreprocessed 28x28 pixel array saved to: {output_path}")
        if confidence < 60:
            print("Warning: Low confidence. Try a clearer image with one centered digit.")
    except Exception as exc:
        raise SystemExit(f"Could not process image: {exc}")

if __name__ == "__main__":
    main()
