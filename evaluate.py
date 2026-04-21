from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = Path(__file__).parent / "models" / "pet_classifier.keras"
RESULTS_DIR = Path(__file__).parent / "results"

IMG_SIZE = 224
BATCH_SIZE = 32


def preprocess(sample):
    image = tf.image.resize(sample["image"], (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, sample["label"]


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run train.py first.")

    ds_test, info = tfds.load(
        "oxford_iiit_pet",
        split="test",
        with_info=True,
        as_supervised=False,
    )
    label_names = info.features["label"].names

    ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(MODEL_PATH)
    loss, acc = model.evaluate(ds_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test loss:     {loss:.4f}\n")

    y_true, y_pred = [], []
    for images, labels in ds_test:
        preds = np.argmax(model.predict(images, verbose=0), axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)

    try:
        import matplotlib.pyplot as plt

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=90, fontsize=7)
        ax.set_yticklabels(label_names, fontsize=7)
        fig.colorbar(im)
        fig.tight_layout()
        out = RESULTS_DIR / "confusion_matrix.png"
        fig.savefig(out, dpi=150)
        print(f"\nSaved confusion matrix to {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
