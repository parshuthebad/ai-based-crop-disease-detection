"""Evaluate a trained model: accuracy, macro F1 and confusion matrix."""
from __future__ import annotations

import argparse
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/PlantVillage")
    parser.add_argument("--model", default="ml/models/crop_model.keras")
    parser.add_argument("--report", default="ml/models/eval_report.json")
    args = parser.parse_args()

    ds = tf.keras.utils.image_dataset_from_directory(
        args.data, validation_split=0.2, subset="validation", seed=1337,
        image_size=IMG_SIZE, batch_size=32, shuffle=False,
    )
    class_names = ds.class_names
    model = tf.keras.models.load_model(args.model)

    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_pred = np.argmax(model.predict(ds, verbose=0), axis=1)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix shape:", confusion_matrix(y_true, y_pred).shape)

    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
