"""Image preprocessing using TensorFlow and NumPy only (no Pillow/OpenCV)."""
import numpy as np
import tensorflow as tf

def preprocess_image_path(image_path):
    """
    Load PNG/JPEG/BMP image using TensorFlow, then prepare one digit
    as a 28x28 MNIST-style tensor. Returns model_input and preview array.
    """
    raw = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(raw, channels=1, expand_animations=False)
    gray = tf.squeeze(image, axis=-1).numpy().astype(np.uint8)

    # Otsu thresholding implemented with NumPy.
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    probability = hist / max(total, 1)
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * np.arange(256))
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    between = np.zeros(256, dtype=np.float64)
    valid = denom > 0
    between[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denom[valid]
    threshold = int(np.argmax(between))

    # Select foreground polarity with fewer foreground pixels, assuming one digit.
    dark_mask = gray <= threshold
    light_mask = gray > threshold
    candidates = [dark_mask, light_mask]
    plausible = [m for m in candidates if 0.002 <= m.mean() <= 0.45]
    if not plausible:
        raise ValueError("Could not isolate a digit. Use a clear, high-contrast image.")
    mask = min(plausible, key=lambda m: abs(m.mean() - 0.18))

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("No digit detected. Please upload a clearer image.")

    # Crop to foreground bounds and preserve aspect ratio in a 20x20 region.
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    crop = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Resize with TensorFlow, avoiding external image libraries.
    resized = tf.image.resize(crop[..., np.newaxis], (new_h, new_w), method="bilinear").numpy()[..., 0]
    resized = np.clip(resized, 0, 255).astype(np.uint8)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized

    # Center foreground intensity by its center of mass.
    mass = canvas.astype(np.float64)
    total_mass = mass.sum()
    if total_mass > 0:
        yy, xx = np.indices(canvas.shape)
        cx = (xx * mass).sum() / total_mass
        cy = (yy * mass).sum() / total_mass
        dx, dy = int(round(13.5 - cx)), int(round(13.5 - cy))
        shifted = np.zeros_like(canvas)
        src_x0, src_x1 = max(0, -dx), min(28, 28-dx)
        src_y0, src_y1 = max(0, -dy), min(28, 28-dy)
        dst_x0, dst_x1 = max(0, dx), min(28, 28+dx)
        dst_y0, dst_y1 = max(0, dy), min(28, 28+dy)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = canvas[src_y0:src_y1, src_x0:src_x1]
        canvas = shifted

    model_input = (canvas.astype("float32") / 255.0).reshape(1, 28, 28, 1)
    return model_input, canvas
