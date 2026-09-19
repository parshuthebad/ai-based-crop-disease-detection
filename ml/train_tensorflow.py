"""Train the crop disease classifier with TensorFlow / Keras (MobileNetV2)."""
from __future__ import annotations

import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)


def build_datasets(data_dir: str, batch_size: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=1337,
        image_size=IMG_SIZE, batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=1337,
        image_size=IMG_SIZE, batch_size=batch_size,
    )
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return (train_ds.prefetch(autotune), val_ds.prefetch(autotune), class_names)


def build_model(num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ])

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/PlantVillage")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="ml/models")
    args = parser.parse_args()

    train_ds, val_ds, class_names = build_datasets(args.data, args.batch_size)
    model = build_model(len(class_names))

    os.makedirs(args.out, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.out, "crop_model.keras"),
            save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    with open(os.path.join(args.out, "class_names.json"), "w", encoding="utf-8") as fh:
        json.dump(class_names, fh, indent=2)
    print(f"Saved model and {len(class_names)} class names to {args.out}")


if __name__ == "__main__":
    main()
