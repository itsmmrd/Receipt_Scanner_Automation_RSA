"""Find a document with the FairScan segmentation model.

The model weights are GPLv3 and are downloaded on first use from
https://github.com/pynicolas/fairscan-segmentation-model
They are not stored in this repository.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://github.com/pynicolas/fairscan-segmentation-model/releases/download/"
    "v1.2.0/fairscan-segmentation-model.tflite"
)
MODEL_PATH = Path(__file__).resolve().parent / "data" / "fairscan-segmentation-model.tflite"
INPUT_SIZE = 256

_interpreter = None


def _model_file() -> Path:
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _get_interpreter():
    global _interpreter
    if _interpreter is None:
        from ai_edge_litert.interpreter import Interpreter

        _interpreter = Interpreter(model_path=str(_model_file()))
        _interpreter.allocate_tensors()
    return _interpreter


def segment_document_quad(image: np.ndarray) -> np.ndarray | None:
    """Return four page corners in the original image, or None."""
    height, width = image.shape[:2]
    interpreter = _get_interpreter()
    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32)
    tensor = (tensor - 127.5) / 127.5
    tensor = tensor[None, ...]
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], tensor)
    interpreter.invoke()
    mask = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    mask = np.squeeze(mask)
    binary = (mask >= 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < INPUT_SIZE * INPUT_SIZE * 0.05:
        return None
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    box[:, 0] *= width / INPUT_SIZE
    box[:, 1] *= height / INPUT_SIZE
    return box
