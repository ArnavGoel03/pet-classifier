from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "pet_classifier.keras"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 37
HEAD_EPOCHS = 10
FINETUNE_EPOCHS = 5
SEED = 42


def preprocess(sample):
    image = tf.image.resize(sample["image"], (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, sample["label"]


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label


def load_datasets():
    (ds_train, ds_test), info = tfds.load(
        "oxford_iiit_pet",
        split=["train", "test"],
        with_info=True,
        as_supervised=False,
    )
    ds_train = (
        ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1024, seed=SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_test = (
        ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds_train, ds_test, info


def build_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)

    ds_train, ds_test, info = load_datasets()
    print(f"Classes: {info.features['label'].num_classes}")
    print(f"Train examples: {info.splits['train'].num_examples}")
    print(f"Test examples:  {info.splits['test'].num_examples}")

    model, base = build_model()
    model.summary()

    print("\n=== Stage 1: head training ===")
    model.fit(ds_train, validation_data=ds_test, epochs=HEAD_EPOCHS, verbose=2)

    print("\n=== Stage 2: fine-tuning ===")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(ds_train, validation_data=ds_test, epochs=FINETUNE_EPOCHS, verbose=2)

    test_loss, test_acc = model.evaluate(ds_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
