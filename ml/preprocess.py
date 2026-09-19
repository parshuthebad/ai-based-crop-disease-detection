"""OpenCV preprocessing pipeline for crop leaf images."""
from __future__ import annotations

import cv2
import numpy as np

IMG_SIZE = (224, 224)


def decode_image(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unsupported or corrupted image file")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def enhance(image: np.ndarray) -> np.ndarray:
    """Reduce noise and equalise lighting so field photos match training data."""
    denoised = cv2.bilateralFilter(image, d=7, sigmaColor=60, sigmaSpace=60)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def segment_leaf(image: np.ndarray) -> np.ndarray:
    """Mask out non-green background to focus the model on the leaf."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (10, 30, 30), (95, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if cv2.countNonZero(mask) < 0.05 * mask.size:
        return image
    return cv2.bitwise_and(image, image, mask=mask)


def preprocess(image_bytes: bytes, segment: bool = False) -> np.ndarray:
    image = enhance(decode_image(image_bytes))
    if segment:
        image = segment_leaf(image)
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    tensor = image.astype("float32") / 255.0
    return np.expand_dims(tensor, axis=0)
