import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

MODEL_PATH = Path(__file__).parent / "models" / "pet_classifier.keras"
IMG_SIZE = 224


def load_and_preprocess(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype="float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr[np.newaxis, ...]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py <path-to-image>")

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run train.py first.")

    _, info = tfds.load("oxford_iiit_pet", split="test", with_info=True, as_supervised=False)
    label_names = info.features["label"].names

    model = tf.keras.models.load_model(MODEL_PATH)
    x = load_and_preprocess(image_path)
    probs = model.predict(x, verbose=0)[0]
    top = int(np.argmax(probs))

    print(f"Predicted breed: {label_names[top]}  (confidence: {probs[top]:.2%})")
    ranked = sorted(enumerate(probs), key=lambda p: -p[1])
    print("Top 5:")
    for idx, prob in ranked[:5]:
        print(f"  {label_names[idx]}: {prob:.2%}")


if __name__ == "__main__":
    main()
