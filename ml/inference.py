"""Shared inference layer used by the FastAPI and Flask servers."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import numpy as np

from ml.preprocess import preprocess
from ml.remedies import REMEDIES

MODEL_PATH = os.getenv("MODEL_PATH", "ml/models/crop_model.keras")
CLASS_PATH = os.getenv("CLASS_PATH", "ml/models/class_names.json")


@lru_cache(maxsize=1)
def load_class_names() -> list[str]:
    if not os.path.exists(CLASS_PATH):
        return []
    with open(CLASS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_model():
    import tensorflow as tf  # imported lazily to keep cold start light

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it with ml/train_tensorflow.py"
        )
    return tf.keras.models.load_model(MODEL_PATH)


def split_label(label: str) -> tuple[str, str]:
    crop, _, condition = label.partition("___")
    return crop.replace("_", " ").strip(), (condition or "unknown").replace("_", " ").strip()


def predict(image_bytes: bytes, top_k: int = 3) -> dict[str, Any]:
    tensor = preprocess(image_bytes)
    probabilities = load_model().predict(tensor, verbose=0)[0]
    class_names = load_class_names()

    order = np.argsort(probabilities)[::-1][:top_k]
    top = [
        {"label": class_names[i], "confidence": round(float(probabilities[i]), 4)}
        for i in order
    ]
    crop, condition = split_label(top[0]["label"])
    return {
        "crop": crop,
        "condition": condition,
        "healthy": condition.lower() == "healthy",
        "confidence": top[0]["confidence"],
        "remedy": REMEDIES.get(top[0]["label"], REMEDIES["default"]),
        "alternatives": top[1:],
    }
