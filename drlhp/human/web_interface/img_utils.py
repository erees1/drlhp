import base64
from numpy.typing import NDArray
import numpy as np
import cv2


def numpy_to_base64(img_array: NDArray[np.uint8]) -> str:
    """Convert the NumPy array to JPEG using OpenCV"""
    assert img_array.sum() > 255, "Image should have something in it"

    _, img_encoded = cv2.imencode(".jpg", img_array)

    # Encode the JPEG data in base64 and return
    return base64.b64encode(img_encoded).decode("utf-8")  # type: ignore
